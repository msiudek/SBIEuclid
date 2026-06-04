"""
Compare atlas SFR-M distribution directly to COSMOS-Web observations.

Loads:
- JWST atlas (training data)
- COSMOS-Web master catalog (real galaxies with LePhare SFR)

Plots logSFR vs logM (not sSFR) with:
- Atlas galaxies (colored by redshift)
- Real COSMOS-Web galaxies (overlaid)
- Schreiber+2015 MS as reference
- Running medians by mass bin

This gives confidence before changing SFH priors.

Usage
-----
python examples/compare_sfr_atlas_vs_observations.py \
    --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
    --outdir sbi-logs/sfr_comparison
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
OBS_DIR = ROOT / "obs" / "obs_properties"
CATALOG = OBS_DIR / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"


def parse_args():
    p = argparse.ArgumentParser(description="Compare atlas SFR-M to real observations")
    p.add_argument("--atlas-name", type=str, default="atlas_jwst_50000_Nparam_2.dbatlas",
                   help="Atlas filename in library/")
    p.add_argument("--outdir", type=str, default="sbi-logs/sfr_comparison",
                   help="Output directory for plots")
    return p.parse_args()


def load_atlas(atlas_name):
    """Load atlas HDF5 hickle format."""
    atlas_path = LIB_DIR / atlas_name
    print(f"Loading atlas: {atlas_path}")

    with h5py.File(atlas_path, "r") as f:
        data_group = f["data"]
        atlas = {
            "logM": data_group['"mstar"'][:],
            "logSFR": data_group['"sfr"'][:],
            "z": data_group['"zval"'][:],
        }

    return atlas


def load_observations():
    """Load COSMOS-Web master catalog with real SFR measurements."""
    print(f"Loading COSMOS-Web master catalog: {CATALOG}")
    if not CATALOG.exists():
        print(f"ERROR: Catalog not found at {CATALOG}")
        return None, None

    ref_hdu = Table.read(CATALOG, hdu=2)

    # Find reference columns
    z_col = "zpdf_med" if "zpdf_med" in ref_hdu.colnames else "z"
    sfr_col = "sfr_med" if "sfr_med" in ref_hdu.colnames else None
    mass_col = "mass_med" if "mass_med" in ref_hdu.colnames else None

    if sfr_col is None or mass_col is None:
        print("ERROR: Could not find SFR or mass columns in catalog")
        print(f"Available columns: {ref_hdu.colnames}")
        return None, None

    z_obs = np.array(ref_hdu[z_col], dtype=float)
    # NOTE: sfr_med is ALREADY in log10 space from LePhare! Don't take log again
    logSFR_obs = np.array(ref_hdu[sfr_col], dtype=float)
    # NOTE: mass_med is ALREADY in log10 space from LePhare!
    logM_obs = np.array(ref_hdu[mass_col], dtype=float)

    # Filter valid entries
    valid = np.isfinite(z_obs) & np.isfinite(logSFR_obs) & np.isfinite(logM_obs) & \
            (z_obs > 0) & (logM_obs > 0)

    print(f"  {valid.sum()} valid galaxies with z, M*, SFR")

    return (z_obs[valid], logM_obs[valid], logSFR_obs[valid])


def schreiber_ms(logM, z):
    """Schreiber+2015 main sequence: log(SFR)."""
    return np.log10(0.83) - 0.027 + 0.76 * np.log10(1 + z) + \
           (np.log10(1 + z) - 0.13) * (10 ** (logM - 10.5))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPARE ATLAS SFR-M vs COSMOS-Web OBSERVATIONS")
    print("=" * 70)

    # Load data
    atlas = load_atlas(args.atlas_name)
    obs_data = load_observations()

    if obs_data is None:
        print("ERROR: Could not load observations. Skipping.")
        return

    z_obs, logM_obs, logSFR_obs = obs_data

    # ──────────────────────────────────────────────────────────────────────
    # Plot 1: logSFR vs logM comparison at different z bins
    # ──────────────────────────────────────────────────────────────────────
    z_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("logSFR vs logM*: Atlas vs COSMOS-Web Observations", fontsize=14, fontweight="bold")

    for idx, (z_min, z_max) in enumerate(z_bins):
        ax = axes[idx // 2, idx % 2]

        # Atlas galaxies at this z
        atlas_mask = (atlas["z"] >= z_min) & (atlas["z"] < z_max)
        if np.sum(atlas_mask) > 0:
            ax.scatter(atlas["logM"][atlas_mask], atlas["logSFR"][atlas_mask],
                      alpha=0.1, s=5, color="blue", label=f"Atlas (N={np.sum(atlas_mask)})")

        # Observations at this z
        obs_mask = (z_obs >= z_min) & (z_obs < z_max)
        if np.sum(obs_mask) > 0:
            ax.scatter(logM_obs[obs_mask], logSFR_obs[obs_mask],
                      alpha=0.15, s=10, color="red", label=f"COSMOS-Web (N={np.sum(obs_mask)})")

        # Schreiber+2015 MS
        logM_seq = np.linspace(8.5, 11.5, 100)
        z_mid = (z_min + z_max) / 2
        logSFR_seq = schreiber_ms(logM_seq, z_mid)
        ax.plot(logM_seq, logSFR_seq, "g--", linewidth=2.5, label="Schreiber+15 MS")

        # Running median: atlas
        logM_bins = np.arange(8.5, 11.5, 0.25)
        for i in range(len(logM_bins) - 1):
            mask = atlas_mask & (atlas["logM"] >= logM_bins[i]) & (atlas["logM"] < logM_bins[i + 1])
            if np.sum(mask) > 5:
                med_sfr = np.median(atlas["logSFR"][mask])
                ax.plot(logM_bins[i] + 0.125, med_sfr, "bs", markersize=7)

        # Running median: observations
        for i in range(len(logM_bins) - 1):
            mask = obs_mask & (logM_obs >= logM_bins[i]) & (logM_obs < logM_bins[i + 1])
            if np.sum(mask) > 5:
                med_sfr = np.median(logSFR_obs[mask])
                ax.plot(logM_bins[i] + 0.125, med_sfr, "r^", markersize=7)

        ax.set_xlabel("log(M*) [M☉]", fontsize=11)
        ax.set_ylabel("log(SFR) [M☉/yr]", fontsize=11)
        ax.set_title(f"z ∈ [{z_min}, {z_max})", fontsize=12, fontweight="bold")
        ax.set_xlim(8.5, 11.5)
        ax.set_ylim(-4, 2.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    plot_file = outdir / "sfr_atlas_vs_obs.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot 1 saved: {plot_file}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Plot 2: Offset from main sequence by z and mass
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Δ(logSFR) = Observed - Schreiber+15 Main Sequence", fontsize=13, fontweight="bold")

    # Offset for atlas
    logSFR_ms_atlas = schreiber_ms(atlas["logM"], atlas["z"])
    offset_atlas = atlas["logSFR"] - logSFR_ms_atlas

    ax = axes[0]
    h = ax.hist2d(atlas["logM"], offset_atlas, bins=[50, 50], cmap="RdBu_r", cmin=-1, cmax=1)
    ax.set_xlabel("log(M*) [M☉]", fontsize=11)
    ax.set_ylabel("Δ(logSFR) from MS [dex]", fontsize=11)
    ax.set_title("Atlas Offset from Main Sequence", fontsize=12, fontweight="bold")
    plt.colorbar(h[3], ax=ax, label="N")

    # Offset for observations
    logSFR_ms_obs = schreiber_ms(logM_obs, z_obs)
    offset_obs = logSFR_obs - logSFR_ms_obs

    ax = axes[1]
    h = ax.hist2d(logM_obs, offset_obs, bins=[50, 50], cmap="RdBu_r", cmin=-1, cmax=1)
    ax.set_xlabel("log(M*) [M☉]", fontsize=11)
    ax.set_ylabel("Δ(logSFR) from MS [dex]", fontsize=11)
    ax.set_title("COSMOS-Web Offset from Main Sequence", fontsize=12, fontweight="bold")
    plt.colorbar(h[3], ax=ax, label="N")

    plt.tight_layout()
    plot_file = outdir / "sfr_offset_from_ms.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot 2 saved: {plot_file}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\nAtlas logSFR distribution (all z):")
    print(f"  Mean: {atlas['logSFR'].mean():.2f}")
    print(f"  Median: {np.median(atlas['logSFR']):.2f}")
    print(f"  Std: {atlas['logSFR'].std():.2f}")

    print("\nCOMOS-Web logSFR distribution (all z):")
    print(f"  Mean: {logSFR_obs.mean():.2f}")
    print(f"  Median: {np.median(logSFR_obs):.2f}")
    print(f"  Std: {logSFR_obs.std():.2f}")

    print("\nOffset from Schreiber+2015 MS:")
    print(f"  Atlas median offset: {np.median(offset_atlas):+.2f} dex")
    print(f"  COSMOS-Web median offset: {np.median(offset_obs):+.2f} dex")

    print("\nOffset by z bin (Atlas):")
    for z_min, z_max in z_bins:
        mask = (atlas["z"] >= z_min) & (atlas["z"] < z_max)
        if np.sum(mask) > 0:
            med_offset = np.median(offset_atlas[mask])
            print(f"  z=[{z_min}, {z_max}): {med_offset:+.2f} dex")

    print("\nOffset by z bin (COSMOS-Web):")
    for z_min, z_max in z_bins:
        mask = (z_obs >= z_min) & (z_obs < z_max)
        if np.sum(mask) > 0:
            med_offset = np.median(offset_obs[mask])
            print(f"  z=[{z_min}, {z_max}): {med_offset:+.2f} dex")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("\nIf Atlas offset < 0: Atlas is BELOW main sequence (too quiescent)")
    print("If COSMOS offset > 0: Real galaxies are ABOVE MS (more star-forming than 'average')")
    print("\nDifference tells us how much to shift the prior.")
    print("\n✓ Done")


if __name__ == "__main__":
    main()
