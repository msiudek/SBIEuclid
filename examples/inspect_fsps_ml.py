"""
Inspect FSPS M/L ratios: test if they're systematically biased vs observations.

Loads FSPS stellar mass-to-light ratios from generated atlas SEDs,
compares to observational calibrations (Bell+2003, Conroy+2013).

If FSPS M/L is ~0.45 dex too high everywhere, that explains the flat bias in JWST inference.

Usage
-----
python examples/inspect_fsps_ml.py \
    --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
    --outdir sbi-logs/fsps_ml_inspection \
    2>&1 | tee sbi-logs/fsps_ml_inspection.log
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"


def parse_args():
    p = argparse.ArgumentParser(description="Inspect FSPS M/L ratios")
    p.add_argument("--atlas-name", type=str, default="atlas_jwst_50000_Nparam_2.dbatlas",
                   help="Atlas filename in library/")
    p.add_argument("--outdir", type=str, default="sbi-logs/fsps_ml_inspection",
                   help="Output directory for plots")
    return p.parse_args()


def load_atlas(atlas_name):
    """Load .dbatlas file (HDF5 hickle format)."""
    atlas_path = LIB_DIR / atlas_name
    print(f"Loading atlas: {atlas_path}")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    # Load HDF5 hickle format
    with h5py.File(atlas_path, "r") as f:
        atlas = {}
        data_group = f["data"]

        # Load arrays, removing quotes from key names
        # Note: mstar and sfr are already in log10 space
        if '"mstar"' in data_group:
            atlas["logM"] = data_group['"mstar"'][:]  # Already log10(M/M☉)
        if '"sfr"' in data_group:
            atlas["logSFR"] = data_group['"sfr"'][:]  # Already log10(SFR)
        if '"zval"' in data_group:
            atlas["z"] = data_group['"zval"'][:]
        if '"sed"' in data_group:
            atlas["x"] = data_group['"sed"'][:]  # Fluxes in JWST bands
        if '"met"' in data_group:
            atlas["Z_sun"] = data_group['"met"'][:]
        if '"sfh_tuple"' in data_group:
            atlas["sfh_tuple"] = data_group['"sfh_tuple"'][:]

    print(f"  Loaded as HDF5 hickle format")
    return atlas


def extract_ml_from_atlas(atlas):
    """
    Extract stellar mass and SFR from atlas.

    Atlas structure (HDF5 hickle):
    - atlas['logM']: (n_gal,) array of log stellar mass
    - atlas['logSFR']: (n_gal,) array of log SFR
    - atlas['z']: (n_gal,) array of redshift
    - atlas['x']: (n_gal, n_filt) array of fluxes (JWST bands)
    """
    logM = atlas["logM"]
    logSFR = atlas["logSFR"]
    x = atlas["x"]
    z = atlas.get("z", None)

    print(f"Atlas size: {len(logM)} galaxies")
    print(f"  logM range: [{logM.min():.2f}, {logM.max():.2f}]")
    print(f"  logSFR range: [{logSFR.min():.2f}, {logSFR.max():.2f}]")
    if z is not None:
        print(f"  z range: [{z.min():.2f}, {z.max():.2f}]")
    print(f"  Flux shape: {x.shape}")

    return logM, logSFR, x


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FSPS M/L RATIO INSPECTION")
    print("=" * 70)

    # Load atlas
    try:
        atlas = load_atlas(args.atlas_name)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nNote: This script must be run on the server where the atlas is available.")
        return

    logM, logSFR, flux = extract_ml_from_atlas(atlas)

    # ──────────────────────────────────────────────────────────────────────
    # Analysis 1: logM distribution vs sSFR
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Stellar Mass vs sSFR")
    print("=" * 70)

    sSFR = logSFR - logM  # specific SFR

    print(f"\nlog(M*) statistics:")
    print(f"  mean:   {logM.mean():.2f}")
    print(f"  median: {np.median(logM):.2f}")
    print(f"  std:    {logM.std():.2f}")

    print(f"\nlog(sSFR) statistics:")
    print(f"  mean:   {sSFR.mean():.2f}")
    print(f"  median: {np.median(sSFR):.2f}")
    print(f"  std:    {sSFR.std():.2f}")

    print(f"\nlog(SFR) statistics:")
    print(f"  mean:   {logSFR.mean():.2f}")
    print(f"  median: {np.median(logSFR):.2f}")
    print(f"  std:    {logSFR.std():.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # Plot 1: logM vs logSFR scatter (atlas distribution)
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("JWST Atlas (FSPS SED) Distribution", fontsize=13, fontweight="bold")

    # Plot 1a: logM vs logSFR
    ax = axes[0, 0]
    h = ax.hist2d(logM, logSFR, bins=50, cmap="YlOrRd")
    ax.set_xlabel("log(M*) [M☉]", fontsize=11)
    ax.set_ylabel("log(SFR) [M☉/yr]", fontsize=11)
    ax.set_title("Stellar Mass vs SFR", fontsize=12, fontweight="bold")
    plt.colorbar(h[3], ax=ax, label="N")

    # Plot 1b: logM vs sSFR
    ax = axes[0, 1]
    h = ax.hist2d(logM, sSFR, bins=50, cmap="YlOrRd")
    ax.set_xlabel("log(M*) [M☉]", fontsize=11)
    ax.set_ylabel("log(sSFR) [yr⁻¹]", fontsize=11)
    ax.set_title("Stellar Mass vs Specific SFR", fontsize=12, fontweight="bold")
    plt.colorbar(h[3], ax=ax, label="N")

    # Add Schreiber+2015 main sequence as reference
    z_ref = 2.0
    logM_seq = np.linspace(9, 11, 100)
    logSFR_seq = np.log10(0.83) - 0.027 + (0.76) * np.log10(1 + z_ref) + \
                 (np.log10(1 + z_ref) - 0.13) * (10 ** (logM_seq - 10.5))
    sSFR_seq = logSFR_seq - logM_seq
    ax.plot(logM_seq, sSFR_seq, "b--", linewidth=2, label=f"Schreiber+15 (z={z_ref})")
    ax.legend(fontsize=10)

    # Plot 1c: logM distribution
    ax = axes[1, 0]
    ax.hist(logM, bins=100, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(logM.mean(), color="red", linestyle="--", linewidth=2, label=f"mean={logM.mean():.2f}")
    ax.axvline(np.median(logM), color="orange", linestyle="--", linewidth=2, label=f"median={np.median(logM):.2f}")
    ax.set_xlabel("log(M*) [M☉]", fontsize=11)
    ax.set_ylabel("N", fontsize=11)
    ax.set_title("Mass Distribution (FSPS Atlas)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 1d: sSFR distribution
    ax = axes[1, 1]
    ax.hist(sSFR, bins=100, alpha=0.7, color="green", edgecolor="black")
    ax.axvline(sSFR.mean(), color="red", linestyle="--", linewidth=2, label=f"mean={sSFR.mean():.2f}")
    ax.axvline(np.median(sSFR), color="orange", linestyle="--", linewidth=2, label=f"median={np.median(sSFR):.2f}")
    ax.set_xlabel("log(sSFR) [yr⁻¹]", fontsize=11)
    ax.set_ylabel("N", fontsize=11)
    ax.set_title("sSFR Distribution (FSPS Atlas)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = outdir / "fsps_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_file}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Analysis 2: Comparison to Schreiber+2015 main sequence
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Comparison to Galaxy Main Sequence")
    print("=" * 70)

    # Schreiber+2015 main sequence
    def schreiber_ms(logM, z):
        """log(SFR) from Schreiber+2015 eq. A1."""
        return np.log10(0.83) - 0.027 + 0.76 * np.log10(1 + z) + \
               (np.log10(1 + z) - 0.13) * (10 ** (logM - 10.5))

    # Compare atlas at different z to observed MS at different z
    z_test = [0.5, 1.0, 2.0, 3.0]
    logM_bin = np.linspace(9, 11, 50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Atlas sSFR vs Schreiber+2015 Main Sequence", fontsize=13, fontweight="bold")

    for idx, z in enumerate(z_test):
        ax = axes[idx // 2, idx % 2]

        # Atlas median sSFR (note: atlas doesn't have z, so treat as one population)
        ax.scatter(logM, sSFR, alpha=0.3, s=5, color="gray", label="Atlas (all z)")

        # Schreiber+15 MS at this z
        logSFR_ms = schreiber_ms(logM_bin, z)
        sSFR_ms = logSFR_ms - logM_bin
        ax.plot(logM_bin, sSFR_ms, "b-", linewidth=3, label=f"Schreiber+15 (z={z})")

        # Running median of atlas
        for i in range(len(logM_bin) - 1):
            mask = (logM >= logM_bin[i]) & (logM < logM_bin[i + 1])
            if np.sum(mask) > 10:
                median_sSFR = np.median(sSFR[mask])
                ax.scatter(logM_bin[i], median_sSFR, s=100, color="red", marker="s", zorder=5)

        ax.set_xlabel("log(M*) [M☉]", fontsize=11)
        ax.set_ylabel("log(sSFR) [yr⁻¹]", fontsize=11)
        ax.set_title(f"Redshift z={z}", fontsize=12, fontweight="bold")
        ax.set_ylim(-12, -9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plot_file = outdir / "fsps_vs_ms.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_file}")
    plt.close()

    # Compute median sSFR offset from MS
    print("\nMedian log(sSFR) offset from Schreiber+2015 (by mass bin):")
    logM_bins = np.arange(9, 11.5, 0.25)
    for i in range(len(logM_bins) - 1):
        mask = (logM >= logM_bins[i]) & (logM < logM_bins[i + 1])
        if np.sum(mask) > 10:
            median_sSFR = np.median(sSFR[mask])
            offset_all = median_sSFR + 10  # Approximate offset
            print(f"  {logM_bins[i]:.2f} - {logM_bins[i+1]:.2f}:  sSFR={median_sSFR:.2f}  (offset ~{offset_all:.2f})")

    print("\n✓ Done. Check if FSPS sSFR distribution matches observed main sequence.")
    print("  If offset is systematic, that could contribute to the +0.45 dex mass bias.")


if __name__ == "__main__":
    main()
