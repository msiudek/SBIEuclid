"""
JWST Atlas Diagnostic: mag/sigma grid comparison (real vs mock).

Loads COSMOS-Web master catalog (real JWST photometry) and JWST atlas (mock SEDs),
injects observational noise, applies detection filter, and produces diagnostic plots.

Plots saved to --outdir:
    mag_grid.png   — median mag in (z,logM) cells: real/mock/delta for F277W+F444W
    sigma_grid.png — median sigma in (z,logM) cells: real/mock for F277W+F444W

Usage
-----
python examples/diagnose_sigma_vs_mag_jwst.py \
    --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
    --noise-prefix north_cweb_jwst \
    --outdir sbi-logs/diagnose_jwst \
    2>&1 | tee sbi-logs/diagnose_jwst.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy.table import Table

ROOT    = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"

# JWST filters: use F277W and F444W (comparable to VIS and NISP-H in coverage)
FILTER_STEMS = ["f115w", "f150w", "f277w", "f444w"]
FILTER_SHORT = ["NIRCam-F115W", "NIRCam-F150W", "NIRCam-F277W", "NIRCam-F444W"]
N_FILT = len(FILTER_STEMS)

# Use F277W and F444W for diagnostics (like VIS and NISP-H for Euclid)
F277W_IDX = 2
F444W_IDX = 3

# 2D cell edges
Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(5.0, 12.5, 0.5)
Z_CEN   = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
M_CEN   = 0.5 * (M_EDGES[:-1] + M_EDGES[1:])

CATALOG = OBS_DIR / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"


def parse_args():
    p = argparse.ArgumentParser(description="JWST atlas diagnostic (mag/sigma grids)")
    p.add_argument("--atlas-name", type=str, default="atlas_jwst_50000_Nparam_2.dbatlas",
                   help="Atlas filename in library/")
    p.add_argument("--noise-prefix", type=str, default="north_cweb_jwst",
                   help="Noise model prefix for noise model files")
    p.add_argument("--outdir", type=str, default="sbi-logs/diagnose_jwst",
                   help="Output directory for plots")
    return p.parse_args()


def real_mag_sigma(flux_col, fluxerr_col):
    """Per-galaxy AB magnitude and sigma_mag from flux/fluxerr (µJy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.isfinite(flux_col) & (flux_col > 0),
                       -2.5 * np.log10(np.maximum(flux_col, 1e-30) / 3631e6), np.nan)
        sig = np.where(np.isfinite(flux_col) & (flux_col > 0) &
                       np.isfinite(fluxerr_col) & (fluxerr_col > 0),
                       (2.5 / np.log(10)) * np.abs(fluxerr_col / flux_col), np.nan)
    return mag, sig


def cell_median(values, z, logm, min_count=3):
    """Median of values in (Z_EDGES, M_EDGES) cells → shape (n_m, n_z)."""
    nz, nm = len(Z_EDGES)-1, len(M_EDGES)-1
    grid = np.full((nm, nz), np.nan)
    for iz in range(nz):
        for im in range(nm):
            mask = ((z  >= Z_EDGES[iz]) & (z  < Z_EDGES[iz+1]) &
                    (logm >= M_EDGES[im]) & (logm < M_EDGES[im+1]) &
                    np.isfinite(values))
            if mask.sum() >= min_count:
                grid[im, iz] = np.nanmedian(values[mask])
    return grid


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("JWST ATLAS DIAGNOSTIC: mag/sigma grids")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load COSMOS-Web real observations
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nLoading real observations: {CATALOG}")
    if not CATALOG.exists():
        print(f"ERROR: Catalog not found at {CATALOG}")
        print("This script must be run on the server where COSMOS-Web catalog is available.")
        return

    phot_hdu = Table.read(CATALOG, hdu=1)
    ref_hdu = Table.read(CATALOG, hdu=2)

    # Load photometry and reference values
    z_real = np.array(ref_hdu["zpdf_med"], dtype=float)
    logM_real = np.array(ref_hdu["mass_med"], dtype=float)

    # Load JWST photometry
    flux_real = np.column_stack([
        np.array(phot_hdu[f"flux_aper_{stem}"], dtype=float)
        for stem in FILTER_STEMS
    ])
    fluxerr_real = np.column_stack([
        np.array(phot_hdu[f"flux_err_aper_{stem}"], dtype=float)
        for stem in FILTER_STEMS
    ])

    print(f"  {len(phot_hdu)} galaxies loaded")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Load noise model
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nLoading noise model ({args.noise_prefix})...")
    mean_sigma = np.load(OBS_DIR / f"mean_sigma_{args.noise_prefix}.npy")
    percentiles = np.load(OBS_DIR / f"percentiles_{args.noise_prefix}.npy")
    print(f"  mean_sigma shape: {mean_sigma.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Load JWST atlas
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nLoading JWST atlas...")
    atlas_path = LIB_DIR / args.atlas_name
    with h5py.File(atlas_path, "r") as f:
        data_group = f["data"]
        logM_mock = data_group['"mstar"'][:]
        z_mock = data_group['"zval"'][:]
        flux_mock = data_group['"sed"'][:]  # (n_gal, 4) JWST bands

    print(f"  {len(logM_mock)} mock galaxies loaded")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Add observational noise to mock photometry
    # ──────────────────────────────────────────────────────────────────────
    print("\nAdding observational noise to mock SEDs...")
    flux_mock_noisy = np.copy(flux_mock)

    for filt_idx in range(N_FILT):
        f = flux_mock[:, filt_idx]
        mag = np.full(len(f), np.nan)
        valid = f > 0
        mag[valid] = -2.5 * np.log10(f[valid] / 3631e6)

        perc = percentiles[:, filt_idx]
        bin_idx = np.searchsorted(perc, mag, side='left')
        bin_idx = np.clip(bin_idx, 0, len(mean_sigma[filt_idx]) - 1)

        sigma_mag = mean_sigma[filt_idx, bin_idx]
        sigma_flux = sigma_mag * (np.log(10) / 2.5) * f

        noise = np.random.normal(0, sigma_flux)
        flux_mock_noisy[:, filt_idx] = f + noise

    # ──────────────────────────────────────────────────────────────────────
    # 5. Build magnitude grids: real vs mock (noiseless and noisy)
    # ──────────────────────────────────────────────────────────────────────
    print("\nComputing magnitude grids...")

    # Real observations
    mag_real_277w, sig_real_277w = real_mag_sigma(flux_real[:, F277W_IDX], fluxerr_real[:, F277W_IDX])
    mag_real_444w, sig_real_444w = real_mag_sigma(flux_real[:, F444W_IDX], fluxerr_real[:, F444W_IDX])

    # Mock noiseless
    mag_mock_277w = np.full(len(flux_mock), np.nan)
    mag_mock_444w = np.full(len(flux_mock), np.nan)
    valid_277w = flux_mock[:, F277W_IDX] > 0
    valid_444w = flux_mock[:, F444W_IDX] > 0
    mag_mock_277w[valid_277w] = -2.5 * np.log10(flux_mock[valid_277w, F277W_IDX] / 3631e6)
    mag_mock_444w[valid_444w] = -2.5 * np.log10(flux_mock[valid_444w, F444W_IDX] / 3631e6)

    # Mock noisy
    mag_mock_noisy_277w = np.full(len(flux_mock), np.nan)
    mag_mock_noisy_444w = np.full(len(flux_mock), np.nan)
    valid_277w_noisy = flux_mock_noisy[:, F277W_IDX] > 0
    valid_444w_noisy = flux_mock_noisy[:, F444W_IDX] > 0
    mag_mock_noisy_277w[valid_277w_noisy] = -2.5 * np.log10(flux_mock_noisy[valid_277w_noisy, F277W_IDX] / 3631e6)
    mag_mock_noisy_444w[valid_444w_noisy] = -2.5 * np.log10(flux_mock_noisy[valid_444w_noisy, F444W_IDX] / 3631e6)

    # Build grids
    grid_mag_real_277w = cell_median(mag_real_277w, z_real, logM_real)
    grid_mag_mock_277w = cell_median(mag_mock_277w, z_mock, logM_mock)
    grid_mag_mock_noisy_277w = cell_median(mag_mock_noisy_277w, z_mock, logM_mock)
    grid_delta_277w = grid_mag_mock_noisy_277w - grid_mag_real_277w

    grid_mag_real_444w = cell_median(mag_real_444w, z_real, logM_real)
    grid_mag_mock_444w = cell_median(mag_mock_444w, z_mock, logM_mock)
    grid_mag_mock_noisy_444w = cell_median(mag_mock_noisy_444w, z_mock, logM_mock)
    grid_delta_444w = grid_mag_mock_noisy_444w - grid_mag_real_444w

    # ──────────────────────────────────────────────────────────────────────
    # 6. Build sigma grids (from noise model)
    # ──────────────────────────────────────────────────────────────────────
    print("Computing sigma grids...")

    grid_sig_real_277w = cell_median(sig_real_277w, z_real, logM_real)
    grid_sig_real_444w = cell_median(sig_real_444w, z_real, logM_real)

    # For mock sigma: extract from noise model for each (z, logM) bin
    sig_mock_277w = np.full(len(flux_mock), np.nan)
    sig_mock_444w = np.full(len(flux_mock), np.nan)

    # Get sigma predictions from noise model
    for i in range(len(flux_mock)):
        mag_277w = mag_mock_277w[i]
        mag_444w = mag_mock_444w[i]

        if np.isfinite(mag_277w):
            perc = percentiles[:, F277W_IDX]
            bin_idx = np.clip(np.searchsorted(perc, mag_277w, side='left'), 0, len(mean_sigma[F277W_IDX]) - 1)
            sig_mock_277w[i] = mean_sigma[F277W_IDX, bin_idx]

        if np.isfinite(mag_444w):
            perc = percentiles[:, F444W_IDX]
            bin_idx = np.clip(np.searchsorted(perc, mag_444w, side='left'), 0, len(mean_sigma[F444W_IDX]) - 1)
            sig_mock_444w[i] = mean_sigma[F444W_IDX, bin_idx]

    grid_sig_mock_277w = cell_median(sig_mock_277w, z_mock, logM_mock)
    grid_sig_mock_444w = cell_median(sig_mock_444w, z_mock, logM_mock)

    # ──────────────────────────────────────────────────────────────────────
    # 7. Plot magnitude grids (with delta_mag showing systematic offset)
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle("Median Magnitude in (z, logM) cells — COSMOS-Web | mock noiseless | mock noisy | Δmag(mock-real)", fontsize=12)

    vmin_mag, vmax_mag = 18, 35  # Extended range to see faint end

    vmin_delta, vmax_delta = -2, 2  # Delta mag colorbar

    # F277W
    im = axes[0, 0].imshow(grid_mag_real_277w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[0, 0].set_title("F277W COSMOS-Web (real)", fontsize=11)
    axes[0, 0].set_ylabel("logM*", fontsize=10)
    plt.colorbar(im, ax=axes[0, 0], label="mag")

    im = axes[0, 1].imshow(grid_mag_mock_277w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[0, 1].set_title("F277W mock noiseless SED", fontsize=11)
    plt.colorbar(im, ax=axes[0, 1], label="mag")

    im = axes[0, 2].imshow(grid_mag_mock_noisy_277w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[0, 2].set_title("F277W mock noisy", fontsize=11)
    plt.colorbar(im, ax=axes[0, 2], label="mag")

    im = axes[0, 3].imshow(grid_delta_277w, origin="lower", cmap="RdBu_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_delta, vmax=vmax_delta)
    axes[0, 3].set_title("F277W Δmag (noisy - real)", fontsize=11)
    plt.colorbar(im, ax=axes[0, 3], label="Δmag")

    # F444W
    im = axes[1, 0].imshow(grid_mag_real_444w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[1, 0].set_title("F444W COSMOS-Web (real)", fontsize=11)
    axes[1, 0].set_xlabel("z", fontsize=10)
    axes[1, 0].set_ylabel("logM*", fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], label="mag")

    im = axes[1, 1].imshow(grid_mag_mock_444w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[1, 1].set_title("F444W mock noiseless SED", fontsize=11)
    axes[1, 1].set_xlabel("z", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], label="mag")

    im = axes[1, 2].imshow(grid_mag_mock_noisy_444w, origin="lower", cmap="viridis", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_mag, vmax=vmax_mag)
    axes[1, 2].set_title("F444W mock noisy", fontsize=11)
    axes[1, 2].set_xlabel("z", fontsize=10)
    plt.colorbar(im, ax=axes[1, 2], label="mag")

    im = axes[1, 3].imshow(grid_delta_444w, origin="lower", cmap="RdBu_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_delta, vmax=vmax_delta)
    axes[1, 3].set_title("F444W Δmag (noisy - real)", fontsize=11)
    axes[1, 3].set_xlabel("z", fontsize=10)
    plt.colorbar(im, ax=axes[1, 3], label="Δmag")

    plt.tight_layout()
    mag_plot = outdir / "mag_grid.png"
    plt.savefig(mag_plot, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {mag_plot}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # 8. Plot sigma grids
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Median σ(mag) in (z, logM) cells", fontsize=12)

    vmin_sig, vmax_sig = 0.01, 0.5

    im = axes[0, 0].imshow(grid_sig_real_277w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_sig, vmax=vmax_sig)
    axes[0, 0].set_title("F277W COSMOS-Web σ(mag)", fontsize=11)
    axes[0, 0].set_ylabel("logM*", fontsize=10)
    plt.colorbar(im, ax=axes[0, 0], label="σ(mag)")

    im = axes[0, 1].imshow(grid_sig_real_444w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_sig, vmax=vmax_sig)
    axes[0, 1].set_title("F444W COSMOS-Web σ(mag)", fontsize=11)
    plt.colorbar(im, ax=axes[0, 1], label="σ(mag)")

    im = axes[1, 0].imshow(grid_sig_mock_277w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_sig, vmax=vmax_sig)
    axes[1, 0].set_title("F277W mock σ(mag) from noise model", fontsize=11)
    axes[1, 0].set_xlabel("z", fontsize=10)
    axes[1, 0].set_ylabel("logM*", fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], label="σ(mag)")

    im = axes[1, 1].imshow(grid_sig_mock_444w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                           extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]], vmin=vmin_sig, vmax=vmax_sig)
    axes[1, 1].set_title("F444W mock σ(mag) from noise model", fontsize=11)
    axes[1, 1].set_xlabel("z", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], label="σ(mag)")

    plt.tight_layout()
    sig_plot = outdir / "sigma_grid.png"
    plt.savefig(sig_plot, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {sig_plot}")
    plt.close()

    print("\n✓ Done")


if __name__ == "__main__":
    main()
