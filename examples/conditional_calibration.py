"""Temporary conditional magnitude calibration layer for debugging.

This module fits and applies an empirical correction\n
    delta_mag(logM, z, band) = median(mock_mag - real_mag)

in (logM, z) bins.

The correction is intended as a short-term calibration layer to test whether
miscalibrated P(flux | M, z) drives inference bias. It is not a physical fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.table import Table


NONDET_MAG = 99.0


def build_phot_col(stem: str, phot_type: str, err: bool = False) -> str:
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"
    return f"{prefix}_{stem}_{phot_type}_aper"


def load_real_detected_mags_from_matched(
    matched_fits: str | Path,
    filter_stems: Iterable[str],
    phot_type: str,
    snr_min: float = 3.0,
    z_col: str = "zfinal",
    logm_col: str = "mass_med",
) -> dict[str, np.ndarray]:
    """Load real detected magnitudes and (logM,z) from matched COSMOS-Web catalog."""
    cat = Table.read(matched_fits)
    filter_stems = list(filter_stems)

    n_filt = len(filter_stems)
    n_gal = len(cat)

    flux = np.full((n_gal, n_filt), np.nan, dtype=float)
    err = np.full((n_gal, n_filt), np.nan, dtype=float)

    for j, stem in enumerate(filter_stems):
        fcol = build_phot_col(stem, phot_type, err=False)
        ecol = build_phot_col(stem, phot_type, err=True)
        if fcol not in cat.colnames or ecol not in cat.colnames:
            continue
        flux[:, j] = np.asarray(cat[fcol], dtype=float)
        err[:, j] = np.asarray(cat[ecol], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux / np.where(err > 0, err, np.nan))

    mag = np.full((n_gal, n_filt), np.nan, dtype=float)
    for j in range(n_filt):
        detected = (
            np.isfinite(flux[:, j])
            & np.isfinite(err[:, j])
            & (err[:, j] > 0)
            & (flux[:, j] > 0)
            & np.isfinite(snr[:, j])
            & (snr[:, j] >= snr_min)
        )
        mag[detected, j] = -2.5 * np.log10(flux[detected, j] / 3631e6)

    return {
        "mag": mag,
        "logm": np.asarray(cat[logm_col], dtype=float),
        "z": np.asarray(cat[z_col], dtype=float),
    }


def _fill_grid_nans(grid: np.ndarray, fallback: float) -> np.ndarray:
    """Fill NaNs in a 2D grid with neighbor medians, then fallback."""
    out = np.array(grid, dtype=float, copy=True)
    if not np.any(np.isfinite(out)):
        return np.full_like(out, fallback, dtype=float)

    for _ in range(8):
        nan_mask = ~np.isfinite(out)
        if not np.any(nan_mask):
            break
        updated = False
        for i, j in zip(*np.where(nan_mask)):
            i0, i1 = max(0, i - 1), min(out.shape[0], i + 2)
            j0, j1 = max(0, j - 1), min(out.shape[1], j + 2)
            neigh = out[i0:i1, j0:j1]
            finite = neigh[np.isfinite(neigh)]
            if finite.size > 0:
                out[i, j] = float(np.median(finite))
                updated = True
        if not updated:
            break

    out[~np.isfinite(out)] = fallback
    return out


def _smooth_grid_box(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply simple 3x3 box smoothing while preserving finite values everywhere."""
    out = np.array(grid, dtype=float, copy=True)
    for _ in range(max(int(passes), 0)):
        new = np.array(out, copy=True)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                i0, i1 = max(0, i - 1), min(out.shape[0], i + 2)
                j0, j1 = max(0, j - 1), min(out.shape[1], j + 2)
                patch = out[i0:i1, j0:j1]
                finite = patch[np.isfinite(patch)]
                if finite.size > 0:
                    new[i, j] = float(np.mean(finite))
        out = new
    return out


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(edges[:-1], dtype=float) + np.asarray(edges[1:], dtype=float))


def _interp_axis(centers: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper indices and linear weights on a 1D bin-center grid."""
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)

    if centers.size == 1:
        idx = np.zeros(values.shape, dtype=int)
        weight = np.zeros(values.shape, dtype=float)
        return idx, idx, weight

    clipped = np.clip(values, centers[0], centers[-1])
    upper = np.searchsorted(centers, clipped, side="right")
    upper = np.clip(upper, 1, centers.size - 1)
    lower = upper - 1

    c0 = centers[lower]
    c1 = centers[upper]
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(c1 > c0, (clipped - c0) / (c1 - c0), 0.0)
    return lower, upper, np.clip(weight, 0.0, 1.0)


def _interp_grid_bilinear(
    grid: np.ndarray,
    mass_bins: np.ndarray,
    z_bins: np.ndarray,
    logm: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate a smoothed 2D calibration grid with bilinear interpolation."""
    mass_centers = _bin_centers(mass_bins)
    z_centers = _bin_centers(z_bins)

    m0, m1, wm = _interp_axis(mass_centers, logm)
    z0, z1, wz = _interp_axis(z_centers, z)

    g00 = grid[m0, z0]
    g10 = grid[m1, z0]
    g01 = grid[m0, z1]
    g11 = grid[m1, z1]

    return (
        (1.0 - wm) * (1.0 - wz) * g00
        + wm * (1.0 - wz) * g10
        + (1.0 - wm) * wz * g01
        + wm * wz * g11
    )


def fit_conditional_delta_mag(
    mock_mag_filter_major: np.ndarray,
    mock_logm: np.ndarray,
    mock_z: np.ndarray,
    real_mag_gal_major: np.ndarray,
    real_logm: np.ndarray,
    real_z: np.ndarray,
    mass_bins: np.ndarray,
    z_bins: np.ndarray,
    min_count: int = 30,
    smooth_passes: int = 2,
    nondet_mag: float = NONDET_MAG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit delta_mag(logM,z,band) = median(mock - real) on a binned grid.

    Parameters
    ----------
    mock_mag_filter_major : (n_filt, n_mock)
    real_mag_gal_major : (n_real, n_filt)
    """
    n_filt = mock_mag_filter_major.shape[0]
    n_mass = len(mass_bins) - 1
    n_z = len(z_bins) - 1

    delta_grid = np.full((n_filt, n_mass, n_z), np.nan, dtype=float)
    n_real_grid = np.zeros((n_filt, n_mass, n_z), dtype=int)
    n_mock_grid = np.zeros((n_filt, n_mass, n_z), dtype=int)

    for b in range(n_filt):
        for i in range(n_mass):
            for k in range(n_z):
                real_mask = (
                    np.isfinite(real_logm)
                    & np.isfinite(real_z)
                    & np.isfinite(real_mag_gal_major[:, b])
                    & (real_logm >= mass_bins[i])
                    & (real_logm < mass_bins[i + 1])
                    & (real_z >= z_bins[k])
                    & (real_z < z_bins[k + 1])
                )
                mock_mask = (
                    np.isfinite(mock_logm)
                    & np.isfinite(mock_z)
                    & np.isfinite(mock_mag_filter_major[b])
                    & (mock_mag_filter_major[b] < nondet_mag - 0.5)
                    & (mock_logm >= mass_bins[i])
                    & (mock_logm < mass_bins[i + 1])
                    & (mock_z >= z_bins[k])
                    & (mock_z < z_bins[k + 1])
                )

                n_r = int(np.sum(real_mask))
                n_m = int(np.sum(mock_mask))
                n_real_grid[b, i, k] = n_r
                n_mock_grid[b, i, k] = n_m

                if n_r >= min_count and n_m >= min_count:
                    real_med = float(np.nanmedian(real_mag_gal_major[real_mask, b]))
                    mock_med = float(np.nanmedian(mock_mag_filter_major[b, mock_mask]))
                    delta_grid[b, i, k] = mock_med - real_med

        # Fill/smooth each band grid to avoid holes and hard bin transitions.
        band_grid = delta_grid[b]
        fallback = float(np.nanmedian(band_grid[np.isfinite(band_grid)])) if np.any(np.isfinite(band_grid)) else 0.0
        band_grid = _fill_grid_nans(band_grid, fallback=fallback)
        band_grid = _smooth_grid_box(band_grid, passes=smooth_passes)
        delta_grid[b] = band_grid

    return delta_grid, n_real_grid, n_mock_grid


def apply_conditional_delta_mag(
    mock_mag_filter_major: np.ndarray,
    mock_logm: np.ndarray,
    mock_z: np.ndarray,
    delta_grid: np.ndarray,
    mass_bins: np.ndarray,
    z_bins: np.ndarray,
    nondet_mag: float = NONDET_MAG,
    interpolation: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conditional delta mag to mock magnitudes.

    Uses mag-domain equivalent of flux correction:
        flux_corrected = flux * 10**(0.4 * delta)
        mag_corrected = mag - delta

    Parameters
    ----------
    interpolation
        "bilinear" evaluates the smoothed grid continuously at bin centers.
        "nearest" reproduces the original piecewise-binned behavior.
    """
    corrected = np.array(mock_mag_filter_major, copy=True, dtype=float)
    applied_delta = np.zeros_like(corrected, dtype=float)

    if interpolation not in {"bilinear", "nearest"}:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    mass_idx = np.digitize(mock_logm, mass_bins) - 1
    z_idx = np.digitize(mock_z, z_bins) - 1
    mass_idx = np.clip(mass_idx, 0, len(mass_bins) - 2)
    z_idx = np.clip(z_idx, 0, len(z_bins) - 2)

    for b in range(corrected.shape[0]):
        if interpolation == "nearest":
            deltas = delta_grid[b, mass_idx, z_idx]
        else:
            deltas = _interp_grid_bilinear(
                delta_grid[b],
                mass_bins=np.asarray(mass_bins, dtype=float),
                z_bins=np.asarray(z_bins, dtype=float),
                logm=np.asarray(mock_logm, dtype=float),
                z=np.asarray(mock_z, dtype=float),
            )
        det = np.isfinite(corrected[b]) & (corrected[b] < nondet_mag - 0.5)
        corrected[b, det] = corrected[b, det] - deltas[det]
        applied_delta[b, det] = deltas[det]

    return corrected, applied_delta


def save_conditional_calibration(
    file_path: str | Path,
    delta_grid: np.ndarray,
    mass_bins: np.ndarray,
    z_bins: np.ndarray,
    filter_names: list[str],
    n_real_grid: np.ndarray | None = None,
    n_mock_grid: np.ndarray | None = None,
    note: str = "temporary empirical calibration layer",
) -> None:
    np.savez(
        file_path,
        delta_grid=np.asarray(delta_grid, dtype=float),
        mass_bins=np.asarray(mass_bins, dtype=float),
        z_bins=np.asarray(z_bins, dtype=float),
        filter_names=np.asarray(filter_names),
        n_real_grid=np.asarray(n_real_grid) if n_real_grid is not None else np.array([]),
        n_mock_grid=np.asarray(n_mock_grid) if n_mock_grid is not None else np.array([]),
        note=np.array([note]),
    )


def load_conditional_calibration(file_path: str | Path) -> dict[str, object]:
    data = np.load(file_path, allow_pickle=True)
    return {
        "delta_grid": np.asarray(data["delta_grid"], dtype=float),
        "mass_bins": np.asarray(data["mass_bins"], dtype=float),
        "z_bins": np.asarray(data["z_bins"], dtype=float),
        "filter_names": [str(x) for x in np.asarray(data["filter_names"])],
    }
