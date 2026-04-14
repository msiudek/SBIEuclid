"""
Stellar mass estimation for real Euclid galaxies matched to COSMOS-Web.

Uses the trained model_euclid_v1.3.pkl (mass_sfr mode, trained with vis_yj2d
mock matching) to infer logM* and logSFR from 10-band photometry in the
matched_euclid_cosmosweb.fits catalog and compares to COSMOS-Web SED masses.

Usage
-----
python examples/inference_cosmosweb.py \
    --n-gal 500 --snr-min 3 --n-bands-min 7 --n-samples 200 \
    --outdir sbi-logs/inference_cosmosweb 2>&1 | tee sbi-logs/inference_cosmosweb.log

Selection criteria applied before inference:
  - valid zfinal photometric redshift
  - valid COSMOS-Web mass_med reference mass
  - at least --n-bands-min filters with SNR >= --snr-min
"""

import argparse
import pickle
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.table import Table
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
OBS_DIR    = ROOT / "obs" / "obs_properties"
LIB_DIR    = ROOT / "library"
CATALOG    = ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"
APERTURE   = "2fwhm"
NOISE_PREFIX = f"north_{APERTURE}"
MODEL_NAME   = "model_euclid_v1.3.pkl"
ATLAS_NAME   = "atlas_obs_euclid_north_validate"

# Filter order from filters_to_use.dat (must match training order)
FILTER_STEMS = [
    "h",            # NISP-H
    "j",            # NISP-J
    "y",            # NISP-Y
    "vis",          # VIS
    "g_ext_hsc",    # HSC-g
    "z_ext_hsc",    # HSC-z
    "g_ext_decam",  # DECam-g
    "r_ext_decam",  # DECam-r
    "i_ext_decam",  # DECam-i
    "z_ext_decam",  # DECam-z
]
FILTER_NAMES = ["NISP-H", "NISP-J", "NISP-Y", "VIS",
                "HSC-g", "HSC-z", "DECam-g", "DECam-r", "DECam-i", "DECam-z"]
N_FILT = len(FILTER_STEMS)


# ── argument parsing ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SBI mass inference on COSMOS-Web matched catalog")
    p.add_argument("--n-gal",       type=int,   default=500,
                   help="Number of galaxies to run inference on (default: 500)")
    p.add_argument("--snr-min",     type=float, default=3.0,
                   help="Minimum SNR per band to count as detected (default: 3)")
    p.add_argument("--n-bands-min", type=int,   default=7,
                   help="Minimum number of bands with SNR >= snr-min (default: 7)")
    p.add_argument("--n-samples",   type=int,   default=200,
                   help="Number of posterior samples per galaxy (default: 200)")
    p.add_argument("--outdir",      type=str,   default="sbi-logs/inference_cosmosweb",
                   help="Output directory for results and plots")
    p.add_argument("--plot-subdir", type=str, default="diagnostics",
                   choices=["diagnostics", "validation_plots", "none"],
                   help="Subdirectory inside outdir for figures (default: diagnostics)")
    p.add_argument("--device",      type=str,   default="cpu",
                   help="Inference device: cpu or cuda (default: cpu)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for galaxy selection (default: 42)")
    return p.parse_args()


# ── photometry helpers ─────────────────────────────────────────────────────

def build_obs_array(flux_2d, fluxerr_2d, limits):
    """
    Convert (n_gal, n_filt) flux/fluxerr (μJy) arrays to the 20-dim
    observation vector expected by the sbipix model:
        [mag_0, sig_0, mag_1, sig_1, ..., mag_9, sig_9]  (interleaved)
    which gets reshaped to (n_gal, 2*n_filt).

    Non-detections (flux < limit or non-finite) get mag=99 and
    mag_sigma = mag(limit).
    """
    n_gal = flux_2d.shape[0]
    obs = np.zeros((n_gal, 2 * N_FILT), dtype=np.float32)

    for j, lim in enumerate(limits):
        f   = flux_2d[:, j]
        fe  = fluxerr_2d[:, j]
        non_detect = ~np.isfinite(f) | (f <= 0) | (f < lim)

        # magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(non_detect, 99.0, -2.5 * np.log10(f / 3631e6))

        # magnitude sigma  (2.5/ln10 * sigma_flux/flux for detections,
        #                   limiting mag for non-detections)
        lim_mag = -2.5 * np.log10(lim / 3631e6) if lim > 0 else 99.0
        with np.errstate(divide='ignore', invalid='ignore'):
            sig = np.where(non_detect,
                           lim_mag,
                           (2.5 / np.log(10)) * np.abs(fe / f))
        # clip absurd errors
        sig = np.clip(sig, 0.001, 5.0)

        obs[:, 2 * j]     = mag
        obs[:, 2 * j + 1] = sig

    return obs


# ── main ──────────────────────────────────────────────────────────────────
def main():
    from sbipix import sbipix

    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir if args.plot_subdir == "none" else (outdir / args.plot_subdir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load catalog and apply selection
    # ------------------------------------------------------------------
    print(f"Loading catalog: {CATALOG}")
    cat = Table.read(CATALOG)
    N_total = len(cat)
    print(f"  {N_total} total matched galaxies")

    # Build flux and fluxerr arrays (n_gal, n_filt) in μJy
    flux    = np.column_stack([np.array(cat[f"flux_{s}_{APERTURE}_aper"],    dtype=float) for s in FILTER_STEMS])
    fluxerr = np.column_stack([np.array(cat[f"fluxerr_{s}_{APERTURE}_aper"], dtype=float) for s in FILTER_STEMS])

    # Reference values
    z_ref    = np.array(cat["zfinal"],   dtype=float)
    mass_ref = np.array(cat["mass_med"], dtype=float)   # log10(M/Msun)
    mass_lo  = np.array(cat["mass_l68"], dtype=float)
    mass_hi  = np.array(cat["mass_u68"], dtype=float)

    # Per-band SNR
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(flux / np.where(fluxerr > 0, fluxerr, np.nan))

    n_bands_snr = np.sum((snr >= args.snr_min) & np.isfinite(snr), axis=1)

    # Selection mask
    has_z    = np.isfinite(z_ref)  & (z_ref > 0)
    has_mass = np.isfinite(mass_ref) & (mass_ref > 5) & (mass_ref < 13)
    has_bands = n_bands_snr >= args.n_bands_min
    good = has_z & has_mass & has_bands

    print(f"  {good.sum()} galaxies pass: z valid + mass valid + ≥{args.n_bands_min} bands SNR≥{args.snr_min}")

    good_idx = np.where(good)[0]
    if len(good_idx) < args.n_gal:
        print(f"  WARNING: only {len(good_idx)} galaxies pass cuts; using all of them")
        sel = good_idx
    else:
        sel = rng.choice(good_idx, size=args.n_gal, replace=False)
    sel = np.sort(sel)
    print(f"  Selected {len(sel)} galaxies for inference")

    flux_sel    = flux[sel]
    fluxerr_sel = fluxerr[sel]
    z_sel       = z_ref[sel]
    mass_sel    = mass_ref[sel]
    mass_lo_sel = mass_lo[sel]
    mass_hi_sel = mass_hi[sel]
    nbands_sel  = n_bands_snr[sel]

    # ------------------------------------------------------------------
    # 2. Configure sbipix model
    # ------------------------------------------------------------------
    print("\nConfiguring sbipix model...")
    sx = sbipix()
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(OBS_DIR),
        mean_sigma_file=f"mean_sigma_{NOISE_PREFIX}.npy",
        std_sigma_file=f"std_sigma_{NOISE_PREFIX}.npy",
        percentiles_file=f"percentiles_{NOISE_PREFIX}.npy",
        limits_file=f"background_noise_{NOISE_PREFIX}.npy",
        lam_eff_file=f"lam_eff_{NOISE_PREFIX}.npy",
    )
    sx.model_path = str(LIB_DIR) + "/"
    sx.model_name = MODEL_NAME
    sx.infer_z    = True   # mass_sfr mode: logM + logSFR both inferred
    sx.include_limit   = True
    sx.include_sigma   = True
    sx.condition_sigma = True
    sx.configure_noise_model(sigma_sampler="mag_lognormal", detection_model="hard")
    sx.load_obs_features()    # populates sx.limits, sx.mean_sigma_obs, sx.percentiles, etc.
    limits = sx.limits        # (n_filt,) in μJy
    print(f"  Flux limits (μJy): {np.array(limits).round(6)}")

    # ------------------------------------------------------------------
    # 3. Build observation array (n_gal, 2*n_filt) - correct mag+magerr
    # ------------------------------------------------------------------
    print("Building observation array...")
    obs = build_obs_array(flux_sel, fluxerr_sel, limits)
    print(f"  obs shape: {obs.shape}  ({len(sel)} galaxies, {2*N_FILT} features)")

    # Quick sanity: print first galaxy
    print(f"  galaxy[0] mags : {obs[0, ::2].round(2)}")
    print(f"  galaxy[0] sigs : {obs[0, 1::2].round(3)}")

    # ------------------------------------------------------------------
    # 4. Load model and run inference
    # ------------------------------------------------------------------
    print(f"\nLoading model: {sx.model_path + sx.model_name}")
    with open(sx.model_path + sx.model_name, "rb") as f:
        qphi = pickle.load(f)

    print(f"Running inference on {len(sel)} galaxies × {args.n_samples} samples ...")
    posteriors = sx._get_posterior_obs(obs, qphi, n_samples=args.n_samples, bar=True, device=args.device)

    # ------------------------------------------------------------------
    # 5. Extract summary statistics
    # ------------------------------------------------------------------
    # theta order for mass_sfr: [0]=logM*, [1]=logSFR
    logM_med  = np.nanmedian(posteriors[:, :, 0], axis=1)
    logM_lo   = np.nanpercentile(posteriors[:, :, 0], 16, axis=1)
    logM_hi   = np.nanpercentile(posteriors[:, :, 0], 84, axis=1)
    logSFR_med = np.nanmedian(posteriors[:, :, 1], axis=1)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    result_file = outdir / "inference_results.npz"
    np.savez(result_file,
             logM_sbi=logM_med, logM_sbi_lo=logM_lo, logM_sbi_hi=logM_hi,
             logSFR_sbi=logSFR_med,
             logM_cosmosweb=mass_sel,
             logM_cosmosweb_lo=mass_lo_sel,
             logM_cosmosweb_hi=mass_hi_sel,
             z=z_sel, n_bands=nbands_sel,
             posteriors=posteriors)
    print(f"\nResults saved to {result_file}")

    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    valid = np.isfinite(logM_med) & np.isfinite(mass_sel)
    delta = logM_med[valid] - mass_sel[valid]
    r, _ = pearsonr(mass_sel[valid], logM_med[valid])
    print(f"\nMass comparison (N={valid.sum()}):")
    print(f"  Pearson r            = {r:.3f}")
    print(f"  Median Δ(SBI-CWeb)   = {np.median(delta):.3f} dex")
    print(f"  NMAD(Δ)              = {1.4826 * np.median(np.abs(delta - np.median(delta))):.3f} dex")
    print(f"  Std(Δ)               = {np.std(delta):.3f} dex")
    print(f"  SBI logM* range      : [{logM_med[valid].min():.2f}, {logM_med[valid].max():.2f}]")
    print(f"  COSMOS-Web logM range: [{mass_sel[valid].min():.2f}, {mass_sel[valid].max():.2f}]")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    # -- 8a. Mass-mass comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    sc = ax.scatter(mass_sel[valid], logM_med[valid],
                    c=z_sel[valid], cmap="plasma_r", vmin=0, vmax=4,
                    s=12, alpha=0.6, linewidths=0, rasterized=True)
    m_range = np.array([mass_sel[valid].min() - 0.3, mass_sel[valid].max() + 0.3])
    ax.plot(m_range, m_range, "k--", lw=1, label="1:1")
    ax.plot(m_range, m_range + np.median(delta), "r-", lw=1,
            label=f"bias = {np.median(delta):+.2f} dex")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("photometric redshift $z$")
    ax.set_xlabel(r"COSMOS-Web $\log M_\star/M_\odot$ (reference)", fontsize=12)
    ax.set_ylabel(r"SBI $\log M_\star/M_\odot$", fontsize=12)
    ax.set_title(f"SBI vs COSMOS-Web stellar mass\n"
                 f"N={valid.sum()}, r={r:.3f}, NMAD={1.4826*np.median(np.abs(delta-np.median(delta))):.3f} dex")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", "box")

    # -- 8b. Residual vs redshift
    ax2 = axes[1]
    sc2 = ax2.scatter(z_sel[valid], delta,
                      c=mass_sel[valid], cmap="viridis", vmin=7, vmax=12,
                      s=12, alpha=0.6, linewidths=0, rasterized=True)
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.axhline(np.median(delta), color="r", lw=1,
                label=f"median = {np.median(delta):+.2f} dex")
    # running median
    z_bins = np.linspace(z_sel[valid].min(), z_sel[valid].max(), 12)
    z_mid = 0.5 * (z_bins[:-1] + z_bins[1:])
    dmed  = [np.median(delta[(z_sel[valid] >= z_bins[k]) & (z_sel[valid] < z_bins[k+1])])
             for k in range(len(z_mid))]
    ax2.plot(z_mid, dmed, "r-o", ms=4, lw=1.5, label="running median")
    cb2 = fig.colorbar(sc2, ax=ax2)
    cb2.set_label(r"COSMOS-Web $\log M_\star$")
    ax2.set_xlabel("photometric redshift $z$", fontsize=12)
    ax2.set_ylabel(r"$\Delta\log M_\star$ (SBI − COSMOS-Web) [dex]", fontsize=12)
    ax2.set_title("Mass residual vs redshift")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-3, 3)

    plt.tight_layout()
    plot_file = plot_dir / "mass_comparison.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_file}")

    # -- 8c. Posterior width vs reference mass
    logM_err = 0.5 * (logM_hi - logM_lo)
    fig2, ax3 = plt.subplots(figsize=(7, 5))
    sc3 = ax3.scatter(mass_sel[valid], logM_err[valid],
                      c=z_sel[valid], cmap="plasma_r", vmin=0, vmax=4,
                      s=12, alpha=0.6, linewidths=0, rasterized=True)
    cb3 = fig2.colorbar(sc3, ax=ax3)
    cb3.set_label("photometric redshift $z$")
    ax3.set_xlabel(r"COSMOS-Web $\log M_\star$ (reference)", fontsize=12)
    ax3.set_ylabel(r"SBI posterior half-width  68% CI [dex]", fontsize=12)
    ax3.set_title("Posterior uncertainty vs reference mass")
    fig2.tight_layout()
    plot_file2 = plot_dir / "posterior_width.png"
    fig2.savefig(plot_file2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved to {plot_file2}")

    # -- 8d. SFR-Mass (star-forming main sequence)
    fig3, ax4 = plt.subplots(figsize=(7, 5))
    sc4 = ax4.scatter(logM_med[valid], logSFR_med[valid],
                      c=z_sel[valid], cmap="plasma_r", vmin=0, vmax=4,
                      s=12, alpha=0.6, linewidths=0, rasterized=True)
    cb4 = fig3.colorbar(sc4, ax=ax4)
    cb4.set_label("photometric redshift $z$")
    ax4.set_xlabel(r"SBI $\log M_\star/M_\odot$", fontsize=12)
    ax4.set_ylabel(r"SBI $\log \mathrm{SFR}\ [M_\odot\,\mathrm{yr}^{-1}]$", fontsize=12)
    ax4.set_title("Star-forming main sequence (SBI estimates)")
    fig3.tight_layout()
    plot_file3 = plot_dir / "sfr_mass.png"
    fig3.savefig(plot_file3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Plot saved to {plot_file3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
