"""
Posterior predictive check for JWST SBI inference.

Two diagnostics:

1. REVERSE CHECK (fast): For each galaxy, compare observed flux vs atlas median
   flux at the *inferred* (logM_sbi, z). If inference is correct, atlas flux at
   inferred mass should match observed flux. Systematic offset = direct M/L bias.

2. FORWARD CHECK (SED reconstruction): For each galaxy, take the inferred
   (logM_sbi, logSFR_sbi, z) and find atlas SEDs at matching parameters.
   Compare reconstructed vs observed SED shape per band.

Both help distinguish:
  - If atlas SEDs systematically don't match observations at the inferred mass
    → confirms the FSPS flux calibration is wrong (flux normalization issue)
  - If atlas SEDs match observations but mass is wrong
    → the SED→mass conversion (M/L) is the problem, not the SED shape

Usage:
    python examples/posterior_predictive_check_jwst.py \
        --inference-dir sbi-logs/inference_cosmosweb_jwst \
        --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
        --outdir sbi-logs/ppc_jwst
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from astropy.table import Table

ROOT     = Path(__file__).resolve().parents[1]
OBS_DIR  = ROOT / "obs" / "obs_properties"
LIB_DIR  = ROOT / "library"
CATALOG  = ROOT / "obs" / "obs_properties" / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"

FILTER_STEMS = ["f115w", "f150w", "f277w", "f444w"]
FILTER_NAMES = ["F115W", "F150W", "F277W", "F444W"]
N_FILT = 4

# (z, logM) grid for cell comparisons
Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(5.0, 13.0, 0.5)


def parse_args():
    p = argparse.ArgumentParser(description="Posterior predictive checks for JWST SBI inference")
    p.add_argument("--inference-dir", default="sbi-logs/inference_cosmosweb_jwst",
                   help="Directory containing inference_results.npz")
    p.add_argument("--atlas-name", default="atlas_jwst_50000_Nparam_2.dbatlas",
                   help="Atlas filename in library/")
    p.add_argument("--outdir", default="sbi-logs/ppc_jwst",
                   help="Output directory for plots")
    p.add_argument("--n-gal", type=int, default=500,
                   help="Number of galaxies to use (default: 500)")
    return p.parse_args()


def load_inference_results(inference_dir):
    r = np.load(Path(inference_dir) / "inference_results.npz", allow_pickle=True)
    return {k: r[k] for k in r.keys()}


def load_catalog_fluxes(selected_indices):
    """Load observed fluxes and errors for selected galaxies from COSMOS-Web."""
    print("Loading COSMOS-Web catalog...")
    cat = Table.read(str(CATALOG), hdu=1)
    fluxes = np.zeros((len(selected_indices), N_FILT), dtype=float)
    fluxerrs = np.zeros((len(selected_indices), N_FILT), dtype=float)
    for j, stem in enumerate(FILTER_STEMS):
        fc  = f"flux_aper_{stem}"
        fec = f"flux_err_aper_{stem}"
        fluxes[:, j]   = np.array(cat[fc][selected_indices],   dtype=float)
        fluxerrs[:, j] = np.array(cat[fec][selected_indices], dtype=float)
    return fluxes, fluxerrs  # µJy


def flux_to_mag(flux_ujy):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux_ujy > 0, -2.5 * np.log10(flux_ujy / 3631e6), np.nan)


def load_atlas(atlas_name):
    atlas_path = LIB_DIR / atlas_name
    print(f"Loading atlas: {atlas_path}")
    data = hickle.load(str(atlas_path))
    logM  = np.array(data["mstar_arr"],   dtype=float)
    z     = np.array(data["zval_arr"],    dtype=float)
    seds  = np.array(data["sed_arr"],     dtype=float)   # (N, N_FILT) µJy
    logSFR = np.array(data["sfr_arr"],    dtype=float)
    return logM, z, seds, logSFR


def cell_median_flux(flux_arr, z_arr, logM_arr):
    """Median flux per (z, logM) cell."""
    nz = len(Z_EDGES) - 1
    nm = len(M_EDGES) - 1
    grid = np.full((nm, nz), np.nan)
    for iz in range(nz):
        for im in range(nm):
            mask = (
                (z_arr    >= Z_EDGES[iz]) & (z_arr    < Z_EDGES[iz+1]) &
                (logM_arr >= M_EDGES[im]) & (logM_arr < M_EDGES[im+1])
            )
            vals = flux_arr[mask]
            if len(vals) > 3:
                grid[im, iz] = np.nanmedian(vals)
    return grid


def get_atlas_flux_at_params(logM_inferred, z_gal, atlas_logM, atlas_z, atlas_seds,
                              dlogM=0.3, dz=0.3):
    """For each galaxy, find atlas SEDs at (logM_inferred ± dlogM, z ± dz) and return median flux."""
    n_gal = len(logM_inferred)
    median_flux = np.full((n_gal, N_FILT), np.nan)
    n_matched = np.zeros(n_gal, dtype=int)

    for i in range(n_gal):
        mask = (
            (np.abs(atlas_logM - logM_inferred[i]) < dlogM) &
            (np.abs(atlas_z    - z_gal[i])         < dz)
        )
        if mask.sum() > 3:
            median_flux[i] = np.nanmedian(atlas_seds[mask], axis=0)
            n_matched[i] = mask.sum()

    print(f"Atlas match: {(n_matched > 0).sum()}/{n_gal} galaxies matched (mean {n_matched[n_matched>0].mean():.0f} atlas SEDs per galaxy)")
    return median_flux


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inference results
    print("Loading inference results...")
    res = load_inference_results(ROOT / args.inference_dir)
    logM_sbi    = res["logM_sbi"].astype(float)
    logSFR_sbi  = res["logSFR_sbi"].astype(float)
    z_gal       = res["z"].astype(float)
    logM_ref    = res["logM_lephare"].astype(float)
    sel_idx     = res["selected_indices"].astype(int)
    n_gal = min(len(logM_sbi), args.n_gal)
    logM_sbi   = logM_sbi[:n_gal]
    logSFR_sbi = logSFR_sbi[:n_gal]
    z_gal      = z_gal[:n_gal]
    logM_ref   = logM_ref[:n_gal]
    sel_idx    = sel_idx[:n_gal]
    delta_logM = logM_sbi - logM_ref

    # Load observed fluxes
    flux_obs, fluxerr_obs = load_catalog_fluxes(sel_idx)  # µJy

    # Load atlas
    atlas_logM, atlas_z, atlas_seds, atlas_logSFR = load_atlas(args.atlas_name)

    # ──────────────────────────────────────────────────────────────────────
    # REVERSE CHECK: atlas flux at inferred mass vs observed flux
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== REVERSE CHECK: atlas flux at inferred mass vs observed ===")
    atlas_flux_at_inferred = get_atlas_flux_at_params(
        logM_sbi, z_gal, atlas_logM, atlas_z, atlas_seds, dlogM=0.3, dz=0.3
    )

    # Also: atlas flux at REFERENCE mass (what we'd expect if inference was perfect)
    atlas_flux_at_reference = get_atlas_flux_at_params(
        logM_ref, z_gal, atlas_logM, atlas_z, atlas_seds, dlogM=0.3, dz=0.3
    )

    # Convert to mag for easier comparison
    mag_obs              = flux_to_mag(flux_obs)
    mag_atlas_inferred   = flux_to_mag(atlas_flux_at_inferred)
    mag_atlas_reference  = flux_to_mag(atlas_flux_at_reference)

    # Residuals: atlas_at_inferred_mass - observed
    # If zero → atlas SEDs at inferred mass perfectly predict observations ✓
    # If negative → atlas too faint at inferred mass → SBI had to infer higher mass
    delta_mag_inferred  = mag_atlas_inferred  - mag_obs  # (n_gal, N_FILT)
    delta_mag_reference = mag_atlas_reference - mag_obs  # (n_gal, N_FILT)

    print("\nMedian Δmag (atlas@inferred_mass - observed) per band:")
    for j, name in enumerate(FILTER_NAMES):
        valid = np.isfinite(delta_mag_inferred[:, j])
        print(f"  {name}: {np.nanmedian(delta_mag_inferred[valid, j]):+.3f} mag "
              f"(σ={np.nanstd(delta_mag_inferred[valid, j]):.3f}, N={valid.sum()})")

    print("\nMedian Δmag (atlas@reference_mass - observed) per band:")
    for j, name in enumerate(FILTER_NAMES):
        valid = np.isfinite(delta_mag_reference[:, j])
        print(f"  {name}: {np.nanmedian(delta_mag_reference[valid, j]):+.3f} mag "
              f"(σ={np.nanstd(delta_mag_reference[valid, j]):.3f}, N={valid.sum()})")

    # ──────────────────────────────────────────────────────────────────────
    # Plot 1: Δmag(atlas@inferred - obs) vs redshift, per band
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Reverse PPC: atlas flux at SBI-inferred mass vs observed\n"
                 "Δmag = 0 → model self-consistent | Δmag ≠ 0 → flux mismatch at inferred mass",
                 fontsize=11)

    for ax, j, fname in zip(axes.flat, range(N_FILT), FILTER_NAMES):
        dm = delta_mag_inferred[:, j]
        valid = np.isfinite(dm)
        sc = ax.scatter(z_gal[valid], dm[valid], c=delta_logM[valid],
                        cmap="RdBu_r", vmin=-2, vmax=2, alpha=0.5, s=10)
        ax.axhline(0, color='k', lw=1, ls='--')
        med = np.nanmedian(dm[valid])
        ax.axhline(med, color='red', lw=1.5, ls='-', label=f"median={med:+.2f}")
        ax.set_xlabel("z")
        ax.set_ylabel("Δmag (atlas@SBI_mass - observed)")
        ax.set_title(fname)
        ax.legend(fontsize=9)
        ax.set_ylim(-3, 3)
        plt.colorbar(sc, ax=ax, label="ΔlogM (SBI - ref)")

    plt.tight_layout()
    out = outdir / "ppc_reverse_delta_mag_vs_z.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {out}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Plot 2: Δmag vs mass (inferred vs reference)
    # ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Reverse PPC: atlas flux at SBI-inferred mass vs observed (vs logM_SBI)",
                 fontsize=11)
    for ax, j, fname in zip(axes.flat, range(N_FILT), FILTER_NAMES):
        dm = delta_mag_inferred[:, j]
        valid = np.isfinite(dm)
        sc = ax.scatter(logM_sbi[valid], dm[valid], c=z_gal[valid],
                        cmap="plasma", vmin=0, vmax=4, alpha=0.5, s=10)
        ax.axhline(0, color='k', lw=1, ls='--')
        med = np.nanmedian(dm[valid])
        ax.axhline(med, color='red', lw=1.5, ls='-', label=f"median={med:+.2f}")
        ax.set_xlabel("SBI logM*")
        ax.set_ylabel("Δmag (atlas@SBI_mass - observed)")
        ax.set_title(fname)
        ax.legend(fontsize=9)
        ax.set_ylim(-3, 3)
        plt.colorbar(sc, ax=ax, label="z")

    plt.tight_layout()
    out = outdir / "ppc_reverse_delta_mag_vs_mass.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Plot 3: SED comparison — observed vs atlas@inferred vs atlas@reference
    # Show 9 representative galaxies in z bins
    # ──────────────────────────────────────────────────────────────────────
    zbins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    fig, axes = plt.subplots(1, len(zbins), figsize=(4 * len(zbins), 4), sharey=False)
    fig.suptitle("SED comparison per z bin: observed (black) | atlas@SBI_mass (red) | atlas@ref_mass (blue)",
                 fontsize=10)

    lam_eff = np.array([1.15, 1.50, 2.77, 4.44])  # µm, approx NIRCam eff wavelengths

    for ax, (zlo, zhi) in zip(axes, zbins):
        mask = (z_gal >= zlo) & (z_gal < zhi)
        if mask.sum() == 0:
            ax.set_title(f"z=[{zlo},{zhi}) N=0")
            continue

        # Pick up to 5 galaxies with good matches
        idx_in_bin = np.where(mask)[0]
        has_match  = np.all(np.isfinite(atlas_flux_at_inferred[idx_in_bin]), axis=1)
        idx_in_bin = idx_in_bin[has_match][:5]
        if len(idx_in_bin) == 0:
            ax.set_title(f"z=[{zlo},{zhi}) no match")
            continue

        for i, gi in enumerate(idx_in_bin):
            alpha = 0.6
            lam_z = lam_eff / (1 + z_gal[gi])  # rest-frame wavelength
            # Observed
            ax.semilogy(lam_z, flux_obs[gi], 'k.-', alpha=alpha)
            # Atlas at SBI mass
            af_infer = atlas_flux_at_inferred[gi]
            if np.all(np.isfinite(af_infer)):
                ax.semilogy(lam_z, af_infer, 'r.-', alpha=alpha)
            # Atlas at reference mass
            af_ref = atlas_flux_at_reference[gi]
            if np.all(np.isfinite(af_ref)):
                ax.semilogy(lam_z, af_ref, 'b.-', alpha=alpha)

        ax.set_xlabel("λ_rest (µm)")
        ax.set_ylabel("Flux (µJy)")
        ax.set_title(f"z=[{zlo},{zhi}) N={len(idx_in_bin)}")

    # legend
    axes[0].plot([], [], 'k.-', label="Observed")
    axes[0].plot([], [], 'r.-', label="Atlas@SBI_mass")
    axes[0].plot([], [], 'b.-', label="Atlas@ref_mass")
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    out = outdir / "ppc_sed_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out}")
    plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Summary printout
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("POSTERIOR PREDICTIVE CHECK SUMMARY")
    print("="*60)
    print(f"\nMass bias (SBI - ref): median = {np.nanmedian(delta_logM):+.3f} dex")
    print(f"\nKey question: At the SBI-inferred mass, does the atlas predict the right flux?")
    print(f"  Δmag ≈ 0   → YES → model self-consistent; bias is in LePhare reference")
    print(f"  Δmag < 0   → Atlas too faint at SBI mass → SBI over-inferred mass to match bright obs")
    print(f"  Δmag > 0   → Atlas too bright at SBI mass → SBI under-inferred mass")
    print()
    print(f"Average Δmag (atlas@SBI_mass - observed), F277W + F444W:")
    for j in [2, 3]:
        valid = np.isfinite(delta_mag_inferred[:, j])
        print(f"  {FILTER_NAMES[j]}: {np.nanmedian(delta_mag_inferred[valid, j]):+.3f} mag")

    print()
    print(f"Average Δmag (atlas@ref_mass - observed), F277W + F444W:")
    for j in [2, 3]:
        valid = np.isfinite(delta_mag_reference[:, j])
        print(f"  {FILTER_NAMES[j]}: {np.nanmedian(delta_mag_reference[valid, j]):+.3f} mag")


if __name__ == "__main__":
    main()
