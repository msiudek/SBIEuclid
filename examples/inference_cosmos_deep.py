from pathlib import Path
import argparse
import numpy as np
from astropy.table import Table


FILTER_SHORT = [
    "NISP-H", "NISP-J", "NISP-Y", "VIS",
    "HSC-g", "HSC-z",
    "DECam-g", "DECam-r", "DECam-i", "DECam-z",
]

FILTER_COL_STEMS = [
    "h", "j", "y", "vis",
    "g_ext_hsc", "z_ext_hsc",
    "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam",
]


def _parse_redshift_columns(raw):
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _select_patch(cat, patch_id):
    if patch_id is None:
        return cat
    patch_col = cat["patch_id_list"]
    try:
        mask_int = patch_col == int(patch_id)
    except (TypeError, ValueError):
        mask_int = np.zeros(len(cat), dtype=bool)
    mask_str = np.array([str(v).strip() == str(patch_id) for v in patch_col])
    return cat[mask_int | mask_str]


def _choose_redshift(cat, redshift_columns, fallback_z):
    for col in redshift_columns:
        if col in cat.colnames:
            z = np.asarray(cat[col], dtype=float)
            return z, col
    if fallback_z is None:
        raise ValueError(
            "No redshift column found. Provide --redshift-columns with an existing column "
            "or set --fallback-z."
        )
    z = np.full(len(cat), float(fallback_z), dtype=float)
    return z, "fallback-z"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SBIPIX posterior inference on COSMOS_DEEP.fits catalog rows"
    )
    parser.add_argument("--fits-file", default="obs/obs_properties/COSMOS_DEEP.fits")
    parser.add_argument("--aperture", default="2fwhm", choices=["2fwhm", "3fwhm"])
    parser.add_argument("--patch-id", type=int, default=98)
    parser.add_argument("--noise-prefix", default="north_2fwhm", choices=["north_2fwhm", "north_3fwhm"])
    parser.add_argument("--model-name", default="post_obs_euclid_north_2fwhm_quick.pkl")
    parser.add_argument("--n-max", type=int, default=5000, help="Max galaxies to infer (<=0 means all)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--min-detected-filters", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Detection threshold for using flux+err directly; lower values are treated as non-detections")
    parser.add_argument("--sample-with", choices=["rejection", "mcmc"], default="mcmc")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--redshift-columns", default="lp_zBEST,photoz,zbest,z_spec,zphot,ez_z_phot")
    parser.add_argument("--fallback-z", type=float, default=None,
                        help="Use fixed redshift if no valid redshift column exists")
    parser.add_argument("--run-name", default="cosmos_catalog_inference")
    return parser


def main():
    args = build_parser().parse_args()

    from sbipix import sbipix

    project_root = Path(__file__).resolve().parents[1]
    obs_dir = project_root / "obs" / "obs_properties"
    logs_dir = project_root / "sbi-logs" / args.run_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    sx = sbipix()
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(obs_dir),
        mean_sigma_file=f"mean_sigma_{args.noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{args.noise_prefix}.npy",
        percentiles_file=f"percentiles_{args.noise_prefix}.npy",
        limits_file=f"background_noise_{args.noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{args.noise_prefix}.npy",
    )

    sx.model_path = str(project_root / "library") + "/"
    sx.model_name = args.model_name

    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False
    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True

    sx.load_obs_features()

    cat = Table.read(args.fits_file)
    cat = _select_patch(cat, args.patch_id)
    if len(cat) == 0:
        raise ValueError(f"No rows found for patch_id={args.patch_id}")

    z_all, z_source = _choose_redshift(cat, _parse_redshift_columns(args.redshift_columns), args.fallback_z)

    n_obj = len(cat) if args.n_max <= 0 else min(args.n_max, len(cat))
    cat = cat[:n_obj]
    z_all = z_all[:n_obj]

    n_filt = len(FILTER_COL_STEMS)
    flux = np.full((n_obj, n_filt), np.nan, dtype=float)
    err = np.full((n_obj, n_filt), np.nan, dtype=float)

    for fi, stem in enumerate(FILTER_COL_STEMS):
        fcol = f"flux_{stem}_{args.aperture}_aper"
        ecol = f"fluxerr_{stem}_{args.aperture}_aper"
        if fcol in cat.colnames:
            flux[:, fi] = np.asarray(cat[fcol], dtype=float)
        if ecol in cat.colnames:
            err[:, fi] = np.asarray(cat[ecol], dtype=float)

    limits = np.asarray(sx.limits, dtype=float)
    phot_arr = flux.copy()
    sigma_arr = err.copy()

    valid = np.isfinite(phot_arr) & np.isfinite(sigma_arr) & (sigma_arr > 0)
    snr = np.where(valid, phot_arr / sigma_arr, np.nan)
    detected = valid & np.isfinite(snr) & (snr > args.snr_min) & (phot_arr > 0)

    nondet = ~detected
    for fi in range(n_filt):
        phot_arr[nondet[:, fi], fi] = 0.5 * limits[fi]
        sigma_arr[nondet[:, fi], fi] = limits[fi]

    n_detected_filters = np.sum(detected, axis=1)
    keep = n_detected_filters >= args.min_detected_filters
    if not np.any(keep):
        raise ValueError(
            f"No galaxies have >= {args.min_detected_filters} detected filters under snr_min={args.snr_min}"
        )

    phot_arr = phot_arr[keep]
    sigma_arr = sigma_arr[keep]
    z_use = z_all[keep]
    n_detected_filters = n_detected_filters[keep]

    print(f"Catalog rows in patch {args.patch_id}: {n_obj}")
    print(f"Kept for inference: {len(phot_arr)} (min detected filters = {args.min_detected_filters})")
    print(f"Redshift source: {z_source}")
    print("Detected-filter percentiles (10/50/90): "
          f"{np.percentile(n_detected_filters, [10, 50, 90]).astype(int)}")

    posteriors = sx.get_posteriors_resolved(
        phot_arr=np.copy(phot_arr),
        n_gal=args.patch_id if args.patch_id is not None else 0,
        n_samples=args.n_samples,
        save=False,
        return_stats=False,
        sigma_arr=np.copy(sigma_arr),
        bar=True,
        input_z=z_use,
        device=args.device,
        sample_with=args.sample_with,
    )

    posterior_median = np.median(posteriors, axis=1)
    posterior_std = np.std(posteriors, axis=1)

    out_file = logs_dir / "cosmos_posteriors.npz"
    np.savez_compressed(
        out_file,
        posteriors=posteriors,
        posterior_median=posterior_median,
        posterior_std=posterior_std,
        redshift=z_use,
        n_detected_filters=n_detected_filters,
        labels=np.array(sx.labels[:posteriors.shape[-1]], dtype=object),
        model_name=np.array([args.model_name], dtype=object),
        sample_with=np.array([args.sample_with], dtype=object),
    )
    print(f"Saved: {out_file}")
    print("Note: if the model was trained with target normalization, outputs remain in model space unless you apply matching de-normalization stats.")


if __name__ == "__main__":
    main()
