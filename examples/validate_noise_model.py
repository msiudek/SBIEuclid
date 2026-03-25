"""
validate_noise_model.py — Pre-training data-space validation.

Runs sbipix simulate + noise injection, loads real COSMOS-Deep photometry
(3fwhm aperture), and produces side-by-side diagnostic plots to confirm
that mock observations match real data within ~10-20% before training.

Plots saved to  sbi-logs/validate_<filter>/  :
  1.  sigma_vs_mag_<filt>.png   — σ vs mag, real scatter + mock scatter + model bins
  2.  mag_hist_<filt>.png       — magnitude histogram, real vs mock (detected only)
  3.  det_fraction_<filt>.png   — detection fraction vs magnitude, real vs mock
  4.  colors.png                — optical/NIR color–color: real vs mock
  5.  sigma_dist_<filt>.png     — distribution of σ values, real vs mock

Usage:
    python examples/validate_noise_model.py \
        --fits-file obs/obs_properties/COSMOS_DEEP.fits \
    --n-sim 10000 --patch-id 98 \
    --selection-band VIS --mag-min 22 --mag-max 28
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Filter metadata — keep in sync with filters_to_use.dat (10 filters)
# ---------------------------------------------------------------------------
FILTER_SHORT = ["NISP-H", "NISP-J", "NISP-Y", "VIS",
                "HSC-g", "HSC-z",
                "DECam-g", "DECam-r", "DECam-i", "DECam-z"]

# FITS column stems (same as FILTER_STEM_TO_COL in learn_obs_noise script)
FILTER_COL_STEMS = ["h", "j", "y", "vis",
                    "g_ext_hsc", "z_ext_hsc",
                    "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]

# Colors to plot: (band_a_idx, band_b_idx, label)
COLOR_PAIRS = [
    (3, 2, "VIS - Y"),
    (2, 1, "Y - J"),
    (1, 0, "J - H"),
    (4, 7, "HSC-g - DECam-r"),
    (5, 9, "HSC-z - DECam-z"),
]

NONDET_MAG = 99.0
SNR_MIN = -5.0
MAG_BRIGHT = 16.0
MAG_FAINT = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flux_ujy_to_mag(flux_ujy):
    """Convert flux in μJy to AB magnitude; returns NaN for non-positive flux."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux_ujy > 0, -2.5 * np.log10(flux_ujy * 1e-6 / 3631.0), np.nan)
    return mag


def flux_ujy_to_mag_err(flux_ujy, fluxerr_ujy):
    """Convert flux error to magnitude error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = np.where(
            flux_ujy > 0,
            np.abs(-2.5 / np.log(10) * fluxerr_ujy / flux_ujy),
            np.nan
        )
    return mag_err


def mag_to_flux_ujy(mag):
    """Convert AB magnitude to flux in μJy."""
    with np.errstate(over="ignore", invalid="ignore"):
        flux = 3631.0 * 1e6 * 10 ** (-0.4 * mag)
    return flux


def band_to_index(band_name):
    """Resolve filter short name to filter index."""
    if band_name is None:
        return None
    lookup = {name.lower(): i for i, name in enumerate(FILTER_SHORT)}
    key = band_name.lower()
    if key not in lookup:
        raise ValueError(f"Unknown band {band_name!r}. Choose from: {', '.join(FILTER_SHORT)}")
    return lookup[key]


def slice_filter_major(data, mask):
    """Apply an object mask to all filter-major arrays in a dict."""
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == mask.size:
            out[key] = value[:, mask]
        else:
            out[key] = value
    return out


def compute_detection_fraction(x, detected, bins, min_count=25):
    """Compute detection fraction in bins of x."""
    total = np.histogram(x, bins=bins)[0].astype(float)
    det = np.histogram(x[detected], bins=bins)[0].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total >= min_count, det / total, np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, frac, total


def load_real_data(fits_path, patch_id=98, aperture="3fwhm", snr_min=SNR_MIN):
    """
    Load photometry from COSMOS-Deep FITS catalog.

    Returns a dict with filter-major arrays.
    """
    from astropy.table import Table

    print(f"Loading real data from {fits_path}  (patch_id={patch_id}, aperture={aperture})")
    cat = Table.read(fits_path)
    # Patch selection
    patch_col = cat["patch_id_list"]
    try:
        mask = patch_col == int(patch_id)
    except (ValueError, TypeError):
        mask = np.zeros(len(cat), dtype=bool)
    str_mask = np.array([str(v).strip() == str(patch_id) for v in patch_col])
    mask = mask | str_mask
    cat = cat[mask]
    print(f"  {len(cat)} galaxies in patch {patch_id}")

    n_filt = len(FILTER_COL_STEMS)
    n_gal = len(cat)
    real_mag = np.full((n_filt, n_gal), np.nan)
    real_sigma = np.full((n_filt, n_gal), np.nan)
    real_flux = np.full((n_filt, n_gal), np.nan)
    real_err = np.full((n_filt, n_gal), np.nan)
    real_valid = np.zeros((n_filt, n_gal), dtype=bool)

    for fi, stem in enumerate(FILTER_COL_STEMS):
        fcol = f"flux_{stem}_{aperture}_aper"
        ecol = f"fluxerr_{stem}_{aperture}_aper"
        if fcol not in cat.colnames:
            print(f"  WARNING: column {fcol!r} not found — filter {FILTER_SHORT[fi]} skipped")
            continue
        flux = np.asarray(cat[fcol], dtype=float)
        err = np.asarray(cat[ecol], dtype=float) if ecol in cat.colnames else np.full(n_gal, np.nan)

        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        snr = np.where(valid, flux / err, np.nan)
        detected = valid & np.isfinite(snr) & (snr > snr_min) & (flux > 0)

        real_flux[fi] = flux
        real_err[fi] = err
        real_valid[fi] = valid
        real_mag[fi] = np.where(detected, flux_ujy_to_mag(flux), np.nan)
        real_sigma[fi] = np.where(detected, flux_ujy_to_mag_err(flux, err), np.nan)

    real_det = np.isfinite(real_mag) & np.isfinite(real_sigma)
    print(f"  Detection fractions: " +
          ", ".join(f"{FILTER_SHORT[i]}={real_det[i].mean():.2f}" for i in range(n_filt)))
    return {
        "mag": real_mag,
        "sigma": real_sigma,
        "det": real_det,
        "flux": real_flux,
        "err": real_err,
        "valid": real_valid,
    }


def get_mock_arrays(sx):
    """
    Extract mock (mag, sigma) from an sbipix instance after add_noise_nan_limit_all().

    Returns a dict with filter-major arrays.
    """
    # sx.mag shape: (n_sim, n_filt, 2) — [:, :, 0]=mag, [:, :, 1]=sigma
    mock_mag = sx.mag[:, :, 0].T.copy()     # (n_filt, n_sim)
    mock_sigma = sx.mag[:, :, 1].T.copy()   # (n_filt, n_sim)
    mock_det = mock_mag < NONDET_MAG - 0.5
    true_mag = sx.obs.T.copy()
    true_flux = mag_to_flux_ujy(true_mag)
    return {
        "mag": mock_mag,
        "sigma": mock_sigma,
        "det": mock_det,
        "true_mag": true_mag,
        "true_flux": true_flux,
    }


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    })


def plot_sigma_vs_mag(fi, real_mag, real_sigma, mock_mag, mock_sigma,
                      mean_sigma_obs, percentiles, outdir):
    """σ vs magnitude: real scatter vs mock scatter + model step function."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f"{FILTER_SHORT[fi]}  —  σ vs magnitude", fontsize=13, fontweight="bold")

    det_r = np.isfinite(real_mag[fi]) & np.isfinite(real_sigma[fi])
    det_m = (mock_mag[fi] < NONDET_MAG - 0.5) & np.isfinite(mock_sigma[fi])

    for ax, lbl, m, s, color in [
        (axes[0], "Real data",        real_mag[fi][det_r],  real_sigma[fi][det_r],  "#1f77b4"),
        (axes[1], "Mock (simulated)", mock_mag[fi][det_m], mock_sigma[fi][det_m], "#ff7f0e"),
    ]:
        ax.scatter(m, s, s=1.5, alpha=0.3, color=color, rasterized=True)

        # Overlay model step function (mean_sigma per bin)
        bins = percentiles[:, fi]
        n_bins = mean_sigma_obs.shape[1]
        edges = np.concatenate([[MAG_BRIGHT], bins, [MAG_FAINT]])
        centers = 0.5 * (edges[:-1] + edges[1:])
        for k in range(n_bins):
            ax.hlines(mean_sigma_obs[fi, k], edges[k], edges[k + 1],
                      colors="red", linewidths=2.0, label="Model mean" if k == 0 else "")
        ax.set_xlim(MAG_BRIGHT, MAG_FAINT)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Magnitude (AB)")
        ax.set_ylabel("σ (mag)")
        ax.set_title(lbl)
        if k == 0:
            ax.legend(loc="upper left")

    # KS-like summary
    if det_r.sum() > 10 and det_m.sum() > 10:
        from scipy.stats import ks_2samp
        s_r = real_sigma[fi][det_r]
        s_m = mock_sigma[fi][det_m]
        s_r = s_r[np.isfinite(s_r)]
        s_m = s_m[np.isfinite(s_m)]
        _, pval = ks_2samp(s_r, s_m)
        fig.text(0.5, 0.01, f"KS p-value (σ distribution): {pval:.3g}",
                 ha="center", fontsize=10, color="grey")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = Path(outdir) / f"sigma_vs_mag_{FILTER_SHORT[fi].replace('/', '-')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_mag_histogram(fi, real_mag, mock_mag, outdir):
    """Magnitude histogram: real vs mock (detected only)."""
    _style()
    det_r = real_mag[fi][np.isfinite(real_mag[fi])]
    det_m = mock_mag[fi][mock_mag[fi] < NONDET_MAG - 0.5]

    if det_r.size == 0 and det_m.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(MAG_BRIGHT, MAG_FAINT, 40)
    ax.hist(det_r, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="Real")
    ax.hist(det_m, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label="Mock")
    ax.set_xlabel("Magnitude (AB)")
    ax.set_ylabel("Density")
    ax.set_title(f"{FILTER_SHORT[fi]}  —  Magnitude distribution (detected)")
    ax.legend()
    plt.tight_layout()
    out = Path(outdir) / f"mag_hist_{FILTER_SHORT[fi].replace('/', '-')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_detection_fraction(fi, real_mag, real_det, mock_mag, mock_det, outdir):
    """Detection fraction vs magnitude bin: real vs mock."""
    _style()
    bins = np.linspace(MAG_BRIGHT, MAG_FAINT, 30)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def det_frac(all_mag, det_mask):
        frac = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (all_mag >= lo) & (all_mag < hi)
            # for real: all_mag includes NaN (non-det → treat as mag=99 for floor)
            # we measure fraction of *all sources we know about in that mag slice*
            # Here: fraction of sources that have a finite mag in this bin
            frac.append(det_mask[in_bin].mean() if in_bin.sum() > 5 else np.nan)
        return np.array(frac)

    # For real: use finite mags as "detected" population; non-finite = non-detected
    all_r = np.isfinite(real_mag[fi])
    all_m = mock_mag[fi] < NONDET_MAG + 0.5  # all (include 99 for binning)

    # Build a combined magnitude for binning including non-detections
    # For real: bin on finite mag where available; we can only compute det-frac
    # in the range where we have both detected and expected sources.
    # Simplification: bin on detected mags only, compare shape of distributions.
    frac_r = det_frac(real_mag[fi][np.isfinite(real_mag[fi])],
                     np.ones(all_r.sum(), dtype=bool))
    frac_m = det_frac(mock_mag[fi][mock_mag[fi] < NONDET_MAG - 0.5],
                     np.ones(det_frac.__code__.co_consts[0] if False else (mock_mag[fi] < NONDET_MAG - 0.5).sum(), dtype=bool))

    # Better approach: compute detection fraction among all simulated galaxies
    # whose *true* SED magnitude falls in each bin
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(centers, frac_r, where="mid", color="#1f77b4", linewidth=2, label="Real (density)")
    ax.step(centers, frac_m, where="mid", color="#ff7f0e", linewidth=2, label="Mock (density)")
    ax.set_xlabel("Magnitude (AB)")
    ax.set_ylabel("Relative density")
    ax.set_title(f"{FILTER_SHORT[fi]}  —  Magnitude distribution shape")
    ax.legend()
    plt.tight_layout()
    out = Path(outdir) / f"mag_shape_{FILTER_SHORT[fi].replace('/', '-')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_detection_fraction_vs_flux(fi, real_data, mock_data, limits, outdir):
    """Detection fraction vs flux using the same limit definition for real and mock."""
    _style()
    limit_flux = limits[fi]
    real_flux = real_data["flux"][fi]
    real_valid = real_data["valid"][fi]
    real_detected = real_valid & (real_flux > limit_flux)

    mock_flux = mock_data["true_flux"][fi]
    mock_detected = mock_data["det"][fi]

    positive_real = real_flux[real_valid & (real_flux > 0)]
    positive_mock = mock_flux[np.isfinite(mock_flux) & (mock_flux > 0)]
    flux_min = np.nanpercentile(np.concatenate([positive_real, positive_mock]), 1)
    flux_max = np.nanpercentile(np.concatenate([positive_real, positive_mock]), 99.5)
    bins = np.geomspace(max(flux_min, 1e-4), max(flux_max, flux_min * 10), 30)

    centers_r, frac_r, total_r = compute_detection_fraction(real_flux[real_valid], real_detected[real_valid], bins)
    centers_m, frac_m, total_m = compute_detection_fraction(mock_flux[np.isfinite(mock_flux)], mock_detected[np.isfinite(mock_flux)], bins)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{FILTER_SHORT[fi]}  —  Detection diagnostics", fontsize=13, fontweight="bold")

    # Left: detection fraction vs flux
    axes[0].step(centers_r, frac_r, where="mid", color="#1f77b4", linewidth=2, label="Real")
    axes[0].step(centers_m, frac_m, where="mid", color="#ff7f0e", linewidth=2, label="Mock")
    axes[0].axvline(limit_flux, color="gray", linestyle="--", linewidth=1, label="Adopted limit")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Flux (μJy)")
    axes[0].set_ylabel("Detection fraction")
    axes[0].set_title("Detection fraction vs flux")
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].legend(loc="lower right")

    # Right: all-source flux distribution around the limit
    bins_hist = np.geomspace(max(flux_min, 1e-4), max(flux_max, flux_min * 10), 40)
    axes[1].hist(real_flux[real_valid], bins=bins_hist, density=True, histtype="step", linewidth=2,
                 color="#1f77b4", label=f"Real all (n={real_valid.sum()})")
    axes[1].hist(mock_flux[np.isfinite(mock_flux)], bins=bins_hist, density=True, histtype="step", linewidth=2,
                 color="#ff7f0e", label=f"Mock all (n={np.isfinite(mock_flux).sum()})")
    axes[1].axvline(limit_flux, color="gray", linestyle="--", linewidth=1, label="Adopted limit")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Flux (μJy)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("All-source flux distribution")
    axes[1].legend()

    plt.tight_layout()
    out = Path(outdir) / f"det_fraction_{FILTER_SHORT[fi].replace('/', '-')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_sigma_distribution(fi, real_sigma, mock_sigma, outdir):
    """Distribution of σ values: real vs mock — checks Normal(mean,std) model."""
    _style()
    det_r = real_sigma[fi][np.isfinite(real_sigma[fi])]
    det_m = mock_sigma[fi][np.isfinite(mock_sigma[fi]) & (mock_sigma[fi] > 0)]
    det_m = det_m[det_m < 2.0]

    if det_r.size < 5 and det_m.size < 5:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1.2, 50)
    ax.hist(det_r, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=f"Real  (n={det_r.size})")
    ax.hist(det_m, bins=bins, density=True, alpha=0.5, color="#ff7f0e", label=f"Mock  (n={det_m.size})")
    ax.set_xlabel("σ (mag)")
    ax.set_ylabel("Density")
    ax.set_title(f"{FILTER_SHORT[fi]}  —  σ distribution (real vs mock)")
    ax.legend()

    # KS test annotation
    if det_r.size > 10 and det_m.size > 10:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(det_r, det_m)
        ax.text(0.97, 0.95, f"KS D={stat:.3f}\np={pval:.3g}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out = Path(outdir) / f"sigma_dist_{FILTER_SHORT[fi].replace('/', '-')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_colors(real_mag, mock_mag, outdir):
    """Color plots with real density background and mock contours."""
    _style()
    n_colors = len(COLOR_PAIRS)
    ncols = min(3, n_colors)
    nrows = (n_colors + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for idx, (ai, bi, label) in enumerate(COLOR_PAIRS):
        ax = axes[idx]
        # Real
        det_r = np.isfinite(real_mag[ai]) & np.isfinite(real_mag[bi])
        color_r = real_mag[ai][det_r] - real_mag[bi][det_r]
        mag_r = real_mag[bi][det_r]
        # Mock
        det_m = (mock_mag[ai] < NONDET_MAG - 0.5) & (mock_mag[bi] < NONDET_MAG - 0.5)
        color_m = mock_mag[ai][det_m] - mock_mag[bi][det_m]
        mag_m = mock_mag[bi][det_m]

        if color_r.size > 0:
            hb = ax.hexbin(mag_r, color_r, gridsize=50, bins="log", mincnt=1,
                           cmap="Blues", linewidths=0)
        if color_m.size > 0:
            hist, xedges, yedges = np.histogram2d(mag_m, color_m, bins=(35, 35), range=((18, 28), (-2, 4)))
            if np.any(hist > 0):
                xcent = 0.5 * (xedges[:-1] + xedges[1:])
                ycent = 0.5 * (yedges[:-1] + yedges[1:])
                levels = np.quantile(hist[hist > 0], [0.5, 0.75, 0.9])
                ax.contour(xcent, ycent, hist.T, levels=np.unique(levels), colors="#ff7f0e", linewidths=1.5)

        ax.set_xlabel(f"{FILTER_SHORT[bi]} (mag)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(-2, 4)
        ax.set_xlim(18, 28)
        legend_els = [
             Line2D([0], [0], color="#1f77b4", linewidth=6, alpha=0.7, label="Real density"),
             Line2D([0], [0], color="#ff7f0e", linewidth=2, label="Mock contours"),
        ]
        ax.legend(handles=legend_els, loc="upper left")

    for idx in range(n_colors, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Color diagnostics: real vs mock", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(outdir) / "colors.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    return out


def plot_sigma_vs_mag_grid(real_mag, real_sigma, mock_mag, mock_sigma,
                           mean_sigma_obs, percentiles, outdir):
    """All-filter overview: σ vs mag grid (2 rows × 5 cols)."""
    _style()
    n_filt = len(FILTER_SHORT)
    ncols = 5
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(28, 9),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.4})

    for fi in range(n_filt):
        row = fi // ncols
        col_base = (fi % ncols) * 2

        for offset, (lbl, m, s, color) in enumerate([
            ("Real",  real_mag[fi],  real_sigma[fi],  "#1f77b4"),
            ("Mock",  mock_mag[fi],  mock_sigma[fi],  "#ff7f0e"),
        ]):
            ax = axes[row, col_base + offset]
            if offset == 0:
                det = np.isfinite(m) & np.isfinite(s)
            else:
                det = (m < NONDET_MAG - 0.5) & np.isfinite(s)
            ax.scatter(m[det], s[det], s=1, alpha=0.2, color=color, rasterized=True)

            # Model step function (real data statistics)
            bins_f = percentiles[:, fi]
            n_bins = mean_sigma_obs.shape[1]
            edges = np.concatenate([[MAG_BRIGHT], bins_f, [MAG_FAINT]])
            for k in range(n_bins):
                ax.hlines(mean_sigma_obs[fi, k], edges[k], edges[k + 1],
                          colors="red", linewidths=1.5)

            ax.set_xlim(MAG_BRIGHT, MAG_FAINT)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{FILTER_SHORT[fi]} — {lbl}", fontsize=8, pad=2)
            if col_base + offset == 0:
                ax.set_ylabel("σ (mag)", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel("Mag", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("σ vs Magnitude — all filters (Real | Mock)  ·  red = model mean",
                 fontsize=12, fontweight="bold")
    out = Path(outdir) / "sigma_vs_mag_ALL.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Pre-training validation: compare mock noise model to real COSMOS data"
    )
    p.add_argument("--fits-file", default="obs/obs_properties/COSMOS_DEEP.fits",
                   help="Path to COSMOS-Deep FITS catalog")
    p.add_argument("--patch-id", type=int, default=98)
    p.add_argument("--aperture", default="3fwhm",
                   choices=["2fwhm", "3fwhm"],
                   help="Aperture to use for real data comparison (default: 3fwhm)")
    p.add_argument("--noise-prefix", default="north_3fwhm",
                   choices=["north_2fwhm", "north_3fwhm"],
                   help="Noise-product prefix to use from obs_properties (default: north_3fwhm)")
    p.add_argument("--limits-file", default=None,
                   help="Optional custom limits file in obs_properties (e.g. background_noise_north_2fwhm_5sigma.npy)")
    p.add_argument("--n-sim", type=int, default=10000,
                   help="Number of mock simulations (default: 10000; use smaller only for local smoke tests)")
    p.add_argument("--outdir", default="sbi-logs/validate",
                   help="Output directory for plots (default: sbi-logs/validate)")
    p.add_argument("--skip-simulate", action="store_true",
                   help="Skip simulation step; load existing atlas instead")
    p.add_argument("--std-scale", type=float, default=1.15,
                   help="Multiply sigma std by this factor before sampling (default: 1.15)")
    p.add_argument("--smooth-bins", action="store_true",
                   help="Interpolate sigma statistics between bins instead of hard digitize")
    p.add_argument("--sigma-sampler", choices=["truncnorm", "lognormal"], default="truncnorm",
                   help="Distribution used to sample sigma values (default: truncnorm)")
    p.add_argument("--sigma-clip-max", type=float, default=1.0,
                   help="Clip sampled sigma values above this threshold in mag (default: 1.0)")
    p.add_argument("--noise-model", choices=["sigma_mag", "depth_corrected"], default="depth_corrected",
                   help="Noise model: classic sigma(mag) or depth-corrected flux model (default: depth_corrected)")
    p.add_argument("--depth-nsigma", type=float, default=1.0,
                   help="Interpret input depth as N-sigma when using depth_corrected (1.0 if limits are 1σ, 5.0 for 5σ)")
    p.add_argument("--corr-clip-min", type=float, default=0.2,
                   help="Minimum correction factor C(m) for depth_corrected mode")
    p.add_argument("--corr-clip-max", type=float, default=5.0,
                   help="Maximum correction factor C(m) for depth_corrected mode")
    p.add_argument("--corr-scatter-log", type=float, default=0.0,
                   help="Optional log-space scatter on C(m); e.g. 0.1 gives mild stochastic spread")
    p.add_argument("--selection-band", choices=FILTER_SHORT, default=None,
                   help="Optional observed-band selection for real and mock catalogs")
    p.add_argument("--mag-min", type=float, default=None,
                   help="Optional lower magnitude cut in the selection band")
    p.add_argument("--mag-max", type=float, default=None,
                   help="Optional upper magnitude cut in the selection band")
    return p


def main():
    args = build_parser().parse_args()

    if args.n_sim < 10000:
        print(f"NOTE: n_sim={args.n_sim} is fine for a local smoke test, but ~10000+ is recommended for validation.")
        print("      Use the server for the final comparison plots.")

    # ------------------------------------------------------------------
    # Setup paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    obs_dir = project_root / "obs" / "obs_properties"
    library_dir = project_root / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fits_path = args.fits_file
    if not os.path.isabs(fits_path):
        fits_path = str(project_root / fits_path)

    # ------------------------------------------------------------------
    # Build mock catalogue via sbipix
    # ------------------------------------------------------------------
    from sbipix import sbipix
    import numpy as np

    sx = sbipix()
    prefix = args.noise_prefix
    limits_file = args.limits_file if args.limits_file is not None else f"background_noise_{prefix}.npy"
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(obs_dir),
        mean_sigma_file=f"mean_sigma_{prefix}.npy",
        std_sigma_file=f"std_sigma_{prefix}.npy",
        percentiles_file=f"percentiles_{prefix}.npy",
        limits_file=limits_file,
        lam_eff_file=f"lam_eff_{prefix}.npy",
    )
    sx.atlas_path = str(library_dir) + "/"
    sx.model_path = str(library_dir) + "/"
    sx.atlas_name = "atlas_obs_euclid_north_validate"
    sx.model_name = "post_obs_euclid_north_validate.pkl"
    sx.n_simulation = args.n_sim
    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False
    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True
    sx.configure_noise_model(
        std_scale=args.std_scale,
        smooth_bins=args.smooth_bins,
        sigma_sampler=args.sigma_sampler,
        sigma_clip_max=args.sigma_clip_max,
        noise_model=args.noise_model,
        depth_nsigma=args.depth_nsigma,
        corr_clip_min=args.corr_clip_min,
        corr_clip_max=args.corr_clip_max,
        corr_scatter_log=args.corr_scatter_log,
    )

    print("Noise-model settings:")
    print(f"  mode           = {sx.noise_model}")
    print(f"  noise prefix   = {prefix}")
    print(f"  limits file    = {limits_file}")
    print(f"  depth nsigma   = {sx.noise_depth_nsigma}")
    print(f"  std scale      = {sx.noise_std_scale}")
    print(f"  smooth bins    = {sx.noise_bin_interpolation}")
    print(f"  sigma sampler  = {sx.noise_sigma_sampler}")
    print(f"  sigma clip max = {sx.noise_sigma_clip_max}")
    print(f"  corr clip      = [{sx.noise_corr_clip_min}, {sx.noise_corr_clip_max}]")
    print(f"  corr scatter   = {sx.noise_corr_scatter_log}")

    np.random.seed(0)

    if args.skip_simulate:
        print("[1/3] Loading existing simulation...")
        sx.load_simulation()
    else:
        print(f"[1/3] Simulating {args.n_sim} galaxy SEDs...")
        sx.simulate(
            mass_min=6.0, mass_max=11.5,
            z_prior="flat", z_min=0.1, z_max=3.0,
            Z_min=-1.7, Z_max=0.3,
            dust_model="Calzetti", dust_prior="flat",
            Av_min=0.0, Av_max=2.5,
        )
        sx.load_simulation()

    print("[2/3] Applying observational realism...")
    sx.load_obs_features()
    sx.add_noise_nan_limit_all()

    # Clean up NaN thetas
    ok = np.isfinite(np.sum(sx.theta, axis=1))
    sx.theta = sx.theta[ok]
    sx.mag = sx.mag[ok]
    sx.obs = sx.obs[ok]
    sx.n_simulation = len(sx.theta)
    print(f"  {sx.n_simulation} valid mock galaxies")

    # ------------------------------------------------------------------
    # Load real data
    # ------------------------------------------------------------------
    print("[3/3] Loading real COSMOS data...")
    real_data = load_real_data(
        fits_path, patch_id=args.patch_id, aperture=args.aperture
    )

    # ------------------------------------------------------------------
    # Extract mock arrays  +  noiseless SED mags
    # ------------------------------------------------------------------
    mock_data = get_mock_arrays(sx)

    # ------------------------------------------------------------------
    # Optional observed-band selection to reduce prior mismatch
    # ------------------------------------------------------------------
    if args.selection_band is not None and (args.mag_min is not None or args.mag_max is not None):
        sel_idx = band_to_index(args.selection_band)
        lo = -np.inf if args.mag_min is None else args.mag_min
        hi = np.inf if args.mag_max is None else args.mag_max

        real_band_mag = real_data["mag"][sel_idx]
        mock_band_mag = mock_data["mag"][sel_idx]

        real_mask = np.isfinite(real_band_mag) & (real_band_mag >= lo) & (real_band_mag <= hi)
        mock_mask = np.isfinite(mock_band_mag) & (mock_band_mag < NONDET_MAG - 0.5) & (mock_band_mag >= lo) & (mock_band_mag <= hi)

        print(f"Applying observed selection in {args.selection_band}: {lo:.1f} <= mag <= {hi:.1f}")
        print(f"  real kept: {real_mask.sum()} / {real_mask.size}")
        print(f"  mock kept: {mock_mask.sum()} / {mock_mask.size}")

        real_data = slice_filter_major(real_data, real_mask)
        mock_data = slice_filter_major(mock_data, mock_mask)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print(f"\nSaving validation plots to {outdir}/")
    saved = []

    # Overview grid
    out = plot_sigma_vs_mag_grid(
        real_data["mag"], real_data["sigma"], mock_data["mag"], mock_data["sigma"],
        sx.mean_sigma_obs, sx.percentiles, outdir
    )
    saved.append(out)
    print(f"  {out.name}")

    # Per-filter detailed plots
    for fi in range(len(FILTER_SHORT)):
        out = plot_sigma_vs_mag(
            fi, real_data["mag"], real_data["sigma"], mock_data["mag"], mock_data["sigma"],
            sx.mean_sigma_obs, sx.percentiles, outdir
        )
        saved.append(out)

        out = plot_mag_histogram(fi, real_data["mag"], mock_data["mag"], outdir)
        if out:
            saved.append(out)

        out = plot_detection_fraction_vs_flux(
            fi, real_data, mock_data, sx.limits, outdir
        )
        if out:
            saved.append(out)

        out = plot_sigma_distribution(fi, real_data["sigma"], mock_data["sigma"], outdir)
        if out:
            saved.append(out)

    print(f"  [per-filter above: sigma_vs_mag, mag_hist, det_fraction, sigma_dist]")

    # Color plots
    out = plot_colors(real_data["mag"], mock_data["mag"], outdir)
    saved.append(out)
    print(f"  {out.name}")

    print(f"\nDone — {len(saved)} plots saved to {outdir}/")
    print("\nAcceptance guideline: real and mock histograms / KDE contours")
    print("should overlap within ~10-20%. If not, check noise files or priors.")


if __name__ == "__main__":
    main()
