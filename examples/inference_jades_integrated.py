"""
Integrated-photometry inference on the JADES catalog (ladder rung 0).

Feeds whole-galaxy integrated fluxes (one catalog row per galaxy) to the SAME
sbipix engine the paper uses per pixel — get_posteriors_resolved. This is the
pixel->integrated port: only the observed-data I/O changes.

    python examples/inference_jades_integrated.py \
        --catalog obs/obs_properties/JADES/hlsp_jades_..._catalog.fits \
        --model-name model_jades_100k.pkl \
        --obs-prefix jades_integrated \
        --aperture kron_conv --n-gal 2000 \
        --outdir sbi-logs/inference_jades_integrated 2>&1 | tee sbi-logs/inf_jades.log

Saves logM_sbi/logSFR_sbi + z/ids (no reference mass — JADES has none; compare
to CIGALE/Prospector separately, and degraded rungs compare per-galaxy by ID).
"""

import argparse
from pathlib import Path

import numpy as np

from jades_catalog import bands_from_filter_list, load_photometry, galaxy_selection

ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--filter-list", default=str(OBS_DIR / "filters_jades_no_wfc.dat"))
    p.add_argument("--model-name", default="model_jades_100k.pkl")
    p.add_argument("--obs-prefix", default="jades_integrated",
                   help="noise-model prefix for sbipix load_obs_features "
                        "(use 'jades_res_bins' for the paper pixel model)")
    p.add_argument("--limits-file", default=None,
                   help="limits npy filename; default background_noise_<obs-prefix>.npy")
    p.add_argument("--aperture", default="kron_conv",
                   choices=["kron_conv", "kron", "circ_conv", "circ"])
    p.add_argument("--circ-index", type=int, default=2)
    p.add_argument("--n-gal", type=int, default=2000)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--snr-min", type=float, default=10.0)
    p.add_argument("--snr-bands", nargs="*", default=["F277W", "F444W"])
    p.add_argument("--z-min", type=float, default=0.2)
    p.add_argument("--z-max", type=float, default=7.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="sbi-logs/inference_jades_integrated")
    return p.parse_args()


def main():
    args = parse_args()
    from sbipix import sbipix

    rng = np.random.default_rng(args.seed)
    bands = bands_from_filter_list(args.filter_list)
    print(f"{len(bands)} bands: {bands}")

    flux, err, meta = load_photometry(args.catalog, bands, aperture=args.aperture,
                                      circ_index=args.circ_index)
    good = galaxy_selection(flux, err, meta, bands, snr_bands=tuple(args.snr_bands),
                            snr_min=args.snr_min, z_min=args.z_min, z_max=args.z_max)
    gidx = np.where(good)[0]
    print(f"{len(gidx)} galaxies pass selection "
          f"(gal, not bright-star, mean SNR[{','.join(args.snr_bands)}]>={args.snr_min}, "
          f"{args.z_min}<z<{args.z_max})")
    if len(gidx) > args.n_gal:
        gidx = np.sort(rng.choice(gidx, size=args.n_gal, replace=False))
    print(f"running inference on {len(gidx)} galaxies")

    flux_sel = flux[gidx].astype(float)
    err_sel = err[gidx].astype(float)
    z_sel = meta["z"][gidx].astype(float)
    # sanitize: non-finite/non-positive flux -> 0 (engine treats as non-detection)
    bad = ~np.isfinite(flux_sel)
    flux_sel[bad] = 0.0
    err_sel[~np.isfinite(err_sel)] = 0.0

    # configure sbipix exactly as the paper's parametric model, integrated noise
    sx = sbipix()
    sx.filter_path = str(OBS_DIR) + "/"
    sx.filter_list = args.filter_list
    sx.model_path = str(LIB_DIR) + "/"
    sx.model_name = args.model_name
    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False
    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True
    sx.obs_prefix = args.obs_prefix
    sx.limits_file = args.limits_file or f"background_noise_{args.obs_prefix}.npy"
    sx.load_obs_features()

    # get_posteriors_resolved mutates phot_arr -> pass copies; per-row z supported
    p = sx.get_posteriors_resolved(
        np.copy(flux_sel), n_gal=0, n_samples=args.n_samples,
        save=False, return_stats=False, sigma_arr=np.copy(err_sel),
        input_z=z_sel, bar=True, device=args.device,
    )
    # p shape (N, n_samples, n_theta); theta=[M*, M*_formed, SFR, tau, ti, M/H, Av]
    logM_sbi = np.median(p[:, :, 0], axis=1)
    logM_lo = np.percentile(p[:, :, 0], 16, axis=1)
    logM_hi = np.percentile(p[:, :, 0], 84, axis=1)
    logSFR_sbi = np.median(p[:, :, 2], axis=1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "inference_results.npz",
             logM_sbi=logM_sbi, logM_sbi_lo=logM_lo, logM_sbi_hi=logM_hi,
             logSFR_sbi=logSFR_sbi, z=z_sel,
             ids=meta["id"][gidx], ra=meta["ra"][gidx], dec=meta["dec"][gidx],
             posteriors=p, selected_indices=gidx)
    print(f"\nlogM_sbi: median={np.median(logM_sbi):.2f} "
          f"[{np.percentile(logM_sbi,5):.2f}, {np.percentile(logM_sbi,95):.2f}]")
    print(f"✓ saved {outdir/'inference_results.npz'}")


if __name__ == "__main__":
    main()
