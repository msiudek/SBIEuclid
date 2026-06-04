"""
JWST atlas diagnostic: noise model and SED coverage in (z, logM*) parameter space.

Loads COSMOS-Web master catalog with LePhare reference and JWST photometry,
injects observational noise via noise model, applies SNR detection filter,
and produces diagnostic grids showing magnitude and σ performance by (z, logM*).

Directly comparable to Euclid diagnose_sigma_vs_mag.py output.

Usage
-----
python examples/diagnose_jwst_atlas.py \
    --outdir sbi-logs/diagnose_jwst \
    2>&1 | tee sbi-logs/diagnose_jwst.log
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.table import Table

ROOT    = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
CATALOG = OBS_DIR / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"

# JWST filters
FILTER_STEMS = ["f115w", "f150w", "f277w", "f444w"]
FILTER_NAMES = ["F115W", "F150W", "F277W", "F444W"]
N_FILT = len(FILTER_STEMS)

# 2D cell edges for mag/sigma grids
Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(5.0, 12.5, 0.5)
Z_CEN   = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
M_CEN   = 0.5 * (M_EDGES[:-1] + M_EDGES[1:])

# z-profile bins (finer, for vs_z plots)
ZP_EDGES = np.arange(0.0, 5.25, 0.25)
ZP_CEN   = 0.5 * (ZP_EDGES[:-1] + ZP_EDGES[1:])


def load_filter_metadata(filter_file, filt_dir):
    """Parse filter list file."""
    entries = []
    with open(os.path.join(filt_dir, filter_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            entries.append({
                "path": os.path.join(filt_dir, parts[0]),
                "short": parts[1],
                "col_stem": parts[2],
            })
    return entries


def parse_args():
    p = argparse.ArgumentParser(description="JWST atlas diagnostic (mag/sigma grids)")
    p.add_argument("--outdir", type=str, default="sbi-logs/diagnose_jwst",
                   help="Output directory for plots")
    p.add_argument("--noise-prefix", type=str, default="cweb_jsst",
                   help="JWST noise model prefix")
    p.add_argument("--snr-min", type=float, default=3.0,
                   help="SNR threshold for detection")
    p.add_argument("--min-det-bands", type=int, default=4,
                   help="Minimum bands with SNR >= snr-min")
    return p.parse_args()


def load_noise_model(noise_prefix, filt_dir=None):
    """Load noise model files."""
    if filt_dir is None:
        filt_dir = OBS_DIR

    mean_sigma = np.load(filt_dir / f"mean_sigma_{noise_prefix}.npy")  # (n_filt, n_bins)
    percentiles = np.load(filt_dir / f"percentiles_{noise_prefix}.npy")  # (n_perc, n_filt)
    return mean_sigma, percentiles


def add_photometric_noise(flux_ujy, flux_err_ujy, mean_sigma_model, percentiles_model):
    """
    Apply observational noise model to flux measurements.

    For each galaxy and filter, determine which magnitude bin it falls into,
    then sample from the noise distribution for that bin.
    """
    n_gal, n_filt = flux_ujy.shape
    flux_noisy = np.copy(flux_ujy)

    for filt_idx in range(n_filt):
        f = flux_ujy[:, filt_idx]
        fe = flux_err_ujy[:, filt_idx]

        # Convert to magnitude
        mag = np.full(n_gal, np.nan)
        valid = (f > 0) & np.isfinite(f)
        mag[valid] = -2.5 * np.log10(f[valid] / 3631e6)

        # Assign galaxies to magnitude bins based on percentiles
        perc = percentiles_model[:, filt_idx]
        bin_idx = np.searchsorted(perc, mag, side='left')
        bin_idx = np.clip(bin_idx, 0, len(mean_sigma_model[filt_idx]) - 1)

        # Add noise: sample from lognormal distribution
        # For simplicity, just use the mean sigma value for each bin
        sigma_mag = mean_sigma_model[filt_idx, bin_idx]

        # Convert mag uncertainty to flux uncertainty
        sigma_flux = sigma_mag * (np.log(10) / 2.5) * f

        # Add Gaussian noise to flux
        noise = np.random.normal(0, sigma_flux)
        flux_noisy[:, filt_idx] = f + noise

    return flux_noisy


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("JWST ATLAS DIAGNOSTIC (mag/sigma grids in z-logM space)")
    print("=" * 70)

    # Load catalog
    print(f"\nLoading COSMOS-Web master catalog: {CATALOG}")
    if not CATALOG.exists():
        print(f"\nERROR: Catalog not found at {CATALOG}")
        print("This script must be run on the server where the catalog is available.")
        return

    phot_hdu = Table.read(CATALOG, hdu=1)
    ref_hdu = Table.read(CATALOG, hdu=2)
    N_gal = min(len(phot_hdu), len(ref_hdu))
    print(f"  {N_gal} galaxies")

    # Load reference data
    z_col = "zpdf_med" if "zpdf_med" in ref_hdu.colnames else "z"
    mass_col = "mass_med" if "mass_med" in ref_hdu.colnames else "logM"
    z_ref = np.array(ref_hdu[z_col][:N_gal], dtype=float)
    mass_ref = np.array(ref_hdu[mass_col][:N_gal], dtype=float)

    # Load photometry
    flux = np.column_stack([
        np.array(phot_hdu[f"flux_aper_{stem}"][:N_gal], dtype=float)
        for stem in FILTER_STEMS
    ])
    fluxerr = np.column_stack([
        np.array(phot_hdu[f"flux_err_aper_{stem}"][:N_gal], dtype=float)
        for stem in FILTER_STEMS
    ])

    print(f"  Photometry shape: {flux.shape}")
    print(f"  z range: [{z_ref[np.isfinite(z_ref)].min():.2f}, {z_ref[np.isfinite(z_ref)].max():.2f}]")
    print(f"  logM range: [{mass_ref[np.isfinite(mass_ref)].min():.2f}, {mass_ref[np.isfinite(mass_ref)].max():.2f}]")

    # Load noise model
    print(f"\nLoading JWST noise model ({args.noise_prefix})...")
    mean_sigma, percentiles = load_noise_model(args.noise_prefix)
    print(f"  mean_sigma shape: {mean_sigma.shape}")

    # Apply SNR detection
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(flux / np.where(fluxerr > 0, fluxerr, np.nan))

    n_det_bands = np.sum((snr >= args.snr_min) & np.isfinite(snr), axis=1)
    detected = n_det_bands >= args.min_det_bands
    print(f"\n  {detected.sum()} galaxies with ≥{args.min_det_bands} bands SNR≥{args.snr_min}")

    # Filter to detected + valid z,M
    valid = detected & np.isfinite(z_ref) & np.isfinite(mass_ref)
    z_valid = z_ref[valid]
    m_valid = mass_ref[valid]
    flux_valid = flux[valid]

    print(f"  {valid.sum()} galaxies with valid z, logM, and detection")

    # ──────────────────────────────────────────────────────────────────────
    # Compute median magnitudes in (z, logM) cells
    # ──────────────────────────────────────────────────────────────────────
    print("\nComputing magnitude grids for F277W and F444W...")

    f277w_idx = 2  # F277W
    f444w_idx = 3  # F444W

    mag_277w = np.full((len(z_valid)), np.nan)
    mag_444w = np.full((len(z_valid)), np.nan)

    valid_277w = flux_valid[:, f277w_idx] > 0
    valid_444w = flux_valid[:, f444w_idx] > 0

    mag_277w[valid_277w] = -2.5 * np.log10(flux_valid[valid_277w, f277w_idx] / 3631e6)
    mag_444w[valid_444w] = -2.5 * np.log10(flux_valid[valid_444w, f444w_idx] / 3631e6)

    # 2D histogram: median magnitude per cell
    mag_grid_277w = np.full((len(M_CEN), len(Z_CEN)), np.nan)
    mag_grid_444w = np.full((len(M_CEN), len(Z_CEN)), np.nan)

    for i_m in range(len(M_CEN)):
        for i_z in range(len(Z_CEN)):
            mask = (z_valid >= Z_EDGES[i_z]) & (z_valid < Z_EDGES[i_z + 1]) & \
                   (m_valid >= M_EDGES[i_m]) & (m_valid < M_EDGES[i_m + 1])

            if np.sum(mask) > 0:
                if np.any(valid_277w[mask]):
                    mag_grid_277w[i_m, i_z] = np.nanmedian(mag_277w[mask])
                if np.any(valid_444w[mask]):
                    mag_grid_444w[i_m, i_z] = np.nanmedian(mag_444w[mask])

    # ──────────────────────────────────────────────────────────────────────
    # Plot magnitude grids
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("JWST Photometry: Median Magnitude in (z, logM) Cells", fontsize=13, fontweight="bold")

    vmin, vmax = 20, 28

    im0 = axes[0].imshow(mag_grid_277w, origin="lower", cmap="viridis", aspect="auto",
                         extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]],
                         vmin=vmin, vmax=vmax)
    axes[0].set_xlabel("Photometric Redshift z", fontsize=11)
    axes[0].set_ylabel("LePhare logM*", fontsize=11)
    axes[0].set_title("F277W Median Magnitude", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="mag")

    im1 = axes[1].imshow(mag_grid_444w, origin="lower", cmap="viridis", aspect="auto",
                         extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]],
                         vmin=vmin, vmax=vmax)
    axes[1].set_xlabel("Photometric Redshift z", fontsize=11)
    axes[1].set_ylabel("LePhare logM*", fontsize=11)
    axes[1].set_title("F444W Median Magnitude", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label="mag")

    plt.tight_layout()
    mag_plot = outdir / "jwst_mag_grid.png"
    plt.savefig(mag_plot, dpi=150, bbox_inches="tight")
    print(f"Saved: {mag_plot}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Compute σ(mag) in (z, logM) cells using noise model
    # ──────────────────────────────────────────────────────────────────────
    print("\nComputing σ(mag) grids from noise model...")

    sigma_grid_277w = np.full((len(M_CEN), len(Z_CEN)), np.nan)
    sigma_grid_444w = np.full((len(M_CEN), len(Z_CEN)), np.nan)

    for i_m in range(len(M_CEN)):
        for i_z in range(len(Z_CEN)):
            mask = (z_valid >= Z_EDGES[i_z]) & (z_valid < Z_EDGES[i_z + 1]) & \
                   (m_valid >= M_EDGES[i_m]) & (m_valid < M_EDGES[i_m + 1])

            if np.sum(mask) > 0:
                # For F277W (filt_idx=2)
                mag_277w_cell = mag_277w[mask]
                valid_277w_cell = np.isfinite(mag_277w_cell)
                if np.any(valid_277w_cell):
                    # Map to nearest percentile bin
                    perc_277w = percentiles[:, f277w_idx]
                    bin_idx = np.searchsorted(perc_277w, mag_277w_cell[valid_277w_cell], side='left')
                    bin_idx = np.clip(bin_idx, 0, len(mean_sigma[f277w_idx]) - 1)
                    sigma_grid_277w[i_m, i_z] = np.median(mean_sigma[f277w_idx, bin_idx])

                # For F444W (filt_idx=3)
                mag_444w_cell = mag_444w[mask]
                valid_444w_cell = np.isfinite(mag_444w_cell)
                if np.any(valid_444w_cell):
                    perc_444w = percentiles[:, f444w_idx]
                    bin_idx = np.searchsorted(perc_444w, mag_444w_cell[valid_444w_cell], side='left')
                    bin_idx = np.clip(bin_idx, 0, len(mean_sigma[f444w_idx]) - 1)
                    sigma_grid_444w[i_m, i_z] = np.median(mean_sigma[f444w_idx, bin_idx])

    # Plot σ grids
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("JWST Noise Model: Median σ(mag) in (z, logM) Cells", fontsize=13, fontweight="bold")

    sigma_vmin, sigma_vmax = 0.01, 0.3

    im0 = axes[0].imshow(sigma_grid_277w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                         extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]],
                         vmin=sigma_vmin, vmax=sigma_vmax)
    axes[0].set_xlabel("Photometric Redshift z", fontsize=11)
    axes[0].set_ylabel("LePhare logM*", fontsize=11)
    axes[0].set_title("F277W σ(mag) from Noise Model", fontsize=12, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="σ_mag (dex)")

    im1 = axes[1].imshow(sigma_grid_444w, origin="lower", cmap="RdYlGn_r", aspect="auto",
                         extent=[Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]],
                         vmin=sigma_vmin, vmax=sigma_vmax)
    axes[1].set_xlabel("Photometric Redshift z", fontsize=11)
    axes[1].set_ylabel("LePhare logM*", fontsize=11)
    axes[1].set_title("F444W σ(mag) from Noise Model", fontsize=12, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label="σ_mag (dex)")

    plt.tight_layout()
    sigma_plot = outdir / "jwst_sigma_grid.png"
    plt.savefig(sigma_plot, dpi=150, bbox_inches="tight")
    print(f"Saved: {sigma_plot}")
    plt.close()

    print("\n✓ Done")


if __name__ == "__main__":
    main()
