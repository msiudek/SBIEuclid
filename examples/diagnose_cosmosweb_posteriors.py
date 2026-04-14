"""
Quick posterior diagnostics for COSMOS-Web inference.

Purpose:
- probe whether posterior sampling issues come from sharp/multimodal posteriors
- summarize posterior width vs redshift and non-detection count
- provide per-galaxy flags for problematic cases

Usage:
python examples/diagnose_cosmosweb_posteriors.py \
  --model-name model_euclid_v1.4.pkl \
  --sample-with mcmc \
  --n-gal 100 --n-samples 400 \
  --outdir sbi-logs/diagnose_cosmosweb_v1.4
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbipix import sbipix


ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"
CATALOG = ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"
APERTURE = "2fwhm"
NOISE_PREFIX = f"north_{APERTURE}"

FILTER_STEMS = [
    "h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
    "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam",
]
N_FILT = len(FILTER_STEMS)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose COSMOS-Web posterior behaviour")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--sample-with", type=str, default="mcmc", choices=["rejection", "mcmc"])
    p.add_argument("--n-gal", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--snr-min", type=float, default=3.0)
    p.add_argument("--n-bands-min", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="sbi-logs/diagnose_cosmosweb")
    return p.parse_args()


def flux_to_mag(flux_ujy):
    return -2.5 * np.log10(flux_ujy / 3631e6)


def build_obs_array(flux_2d, fluxerr_2d, limits):
    n_gal = flux_2d.shape[0]
    obs = np.zeros((n_gal, 2 * N_FILT), dtype=np.float32)
    n_nondet = np.zeros(n_gal, dtype=int)

    for j, lim in enumerate(limits):
        f = flux_2d[:, j]
        fe = fluxerr_2d[:, j]
        non_detect = ~np.isfinite(f) | (f <= 0) | (f < lim)
        n_nondet += non_detect.astype(int)

        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.where(non_detect, 99.0, flux_to_mag(f))

        lim_mag = flux_to_mag(lim) if lim > 0 else 99.0
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = np.where(non_detect, lim_mag, (2.5 / np.log(10)) * np.abs(fe / f))
        sig = np.clip(sig, 0.001, 5.0)

        obs[:, 2 * j] = mag
        obs[:, 2 * j + 1] = sig

    return obs, n_nondet


def count_modes_1d(samples, bins=40, prominence=0.1):
    hist, _ = np.histogram(samples, bins=bins, density=True)
    if not np.isfinite(hist).any() or np.nanmax(hist) <= 0:
        return 0
    h = np.nan_to_num(hist, nan=0.0)
    thr = prominence * np.max(h)
    peaks, _ = find_peaks(h, prominence=thr)
    return len(peaks)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading catalog: {CATALOG}")
    cat = Table.read(CATALOG)

    flux = np.column_stack([np.asarray(cat[f"flux_{s}_{APERTURE}_aper"], dtype=float) for s in FILTER_STEMS])
    fluxerr = np.column_stack([np.asarray(cat[f"fluxerr_{s}_{APERTURE}_aper"], dtype=float) for s in FILTER_STEMS])
    z_ref = np.asarray(cat["zfinal"], dtype=float)
    m_ref = np.asarray(cat["mass_med"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux / np.where(fluxerr > 0, fluxerr, np.nan))
    n_bands = np.sum((snr >= args.snr_min) & np.isfinite(snr), axis=1)

    good = np.isfinite(z_ref) & (z_ref > 0) & np.isfinite(m_ref) & (n_bands >= args.n_bands_min)
    idx = np.where(good)[0]
    if len(idx) == 0:
        raise RuntimeError("No galaxies pass selection.")

    if len(idx) > args.n_gal:
        idx = np.sort(rng.choice(idx, size=args.n_gal, replace=False))

    flux_sel = flux[idx]
    fluxerr_sel = fluxerr[idx]
    z_sel = z_ref[idx]
    m_sel = m_ref[idx]

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
    sx.model_name = args.model_name
    sx.infer_z = False
    sx.include_limit = True
    sx.include_sigma = True
    sx.condition_sigma = True
    sx.configure_noise_model(sigma_sampler="mag_lognormal", detection_model="hard")
    sx.load_obs_features()

    obs, n_nondet = build_obs_array(flux_sel, fluxerr_sel, sx.limits)

    model_file = sx.model_path + sx.model_name
    with open(model_file, "rb") as f:
        qphi = pickle.load(f)

    if args.sample_with != "rejection":
        anpe_file = sx.model_path + "anpe_" + sx.model_name
        with open(anpe_file, "rb") as f:
            anpe = pickle.load(f)
        qphi = anpe.build_posterior(sample_with=args.sample_with)

    try:
        qphi.sample((1,), x=torch.zeros((1, obs.shape[1] + 1), dtype=torch.float32), show_progress_bars=False)
    except Exception as exc:
        raise RuntimeError(
            "Model is incompatible with z-conditioned context (obs+z)."
        ) from exc

    print(f"Sampling {len(obs)} galaxies with backend={args.sample_with}, n_samples={args.n_samples}")
    post = sx._get_posterior_obs(obs, qphi, n_samples=args.n_samples, bar=True, input_z=z_sel, device=args.device)

    logm = post[:, :, 0]
    logsfr = post[:, :, 1]

    logm_med = np.median(logm, axis=1)
    logm_w68 = np.percentile(logm, 84, axis=1) - np.percentile(logm, 16, axis=1)
    logsfr_w68 = np.percentile(logsfr, 84, axis=1) - np.percentile(logsfr, 16, axis=1)

    n_modes_m = np.array([count_modes_1d(logm[i]) for i in range(len(logm))])
    n_modes_s = np.array([count_modes_1d(logsfr[i]) for i in range(len(logsfr))])
    multimodal = (n_modes_m >= 2) | (n_modes_s >= 2)

    delta_m = logm_med - m_sel

    report = np.column_stack([
        idx,
        z_sel,
        m_sel,
        logm_med,
        delta_m,
        logm_w68,
        logsfr_w68,
        n_nondet,
        n_modes_m,
        n_modes_s,
        multimodal.astype(int),
    ])

    hdr = "idx z_ref logM_ref logM_med deltaM logM_w68 logSFR_w68 n_nondet n_modes_logM n_modes_logSFR multimodal_flag"
    np.savetxt(outdir / "posterior_diagnostics.txt", report, header=hdr, fmt="%.6g")
    np.savez(
        outdir / "posterior_diagnostics.npz",
        idx=idx,
        z=z_sel,
        logM_ref=m_sel,
        logM_med=logm_med,
        deltaM=delta_m,
        logM_w68=logm_w68,
        logSFR_w68=logsfr_w68,
        n_nondet=n_nondet,
        n_modes_logM=n_modes_m,
        n_modes_logSFR=n_modes_s,
        multimodal=multimodal,
    )

    valid = np.isfinite(logm_med) & np.isfinite(m_sel)
    if np.sum(valid) > 3:
        r, _ = pearsonr(m_sel[valid], logm_med[valid])
    else:
        r = np.nan

    print(f"N galaxies: {len(idx)}")
    print(f"Median ΔlogM (SBI-CWeb): {np.median(delta_m):.3f} dex")
    print(f"Median width logM (68%): {np.median(logm_w68):.3f} dex")
    print(f"Multimodal fraction: {np.mean(multimodal) * 100:.1f}%")
    print(f"Corr(logM_ref, logM_med): {r:.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(n_nondet, logm_w68, c=z_sel, s=14, alpha=0.7)
    axes[0].set_xlabel("Number of non-detections (10 bands)")
    axes[0].set_ylabel("logM 68% width [dex]")
    axes[0].set_title("Posterior width vs non-detections")

    axes[1].scatter(z_sel, logm_w68, c=multimodal.astype(float), cmap="coolwarm", s=14, alpha=0.7)
    axes[1].set_xlabel("z (catalog)")
    axes[1].set_ylabel("logM 68% width [dex]")
    axes[1].set_title("Posterior width vs z (color=multimodal)")

    fig.tight_layout()
    fig.savefig(outdir / "posterior_width_diagnostics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Example marginal histograms for most problematic galaxies
    rank = np.argsort(logm_w68)[::-1][:6]
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    axes2 = axes2.ravel()
    for k, gi in enumerate(rank):
        ax = axes2[k]
        ax.hist(logm[gi], bins=30, alpha=0.7, color="tab:blue")
        ax.set_title(f"idx={idx[gi]} z={z_sel[gi]:.2f} w68={logm_w68[gi]:.2f}")
        ax.set_xlabel("logM")
    fig2.tight_layout()
    fig2.savefig(outdir / "worst_logM_marginals.png", dpi=140, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved diagnostics to {outdir}")


if __name__ == "__main__":
    main()
