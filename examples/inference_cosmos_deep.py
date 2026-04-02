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


def _choose_redshift(cat, redshift_columns):
    for col in redshift_columns:
        if col in cat.colnames:
            z = np.asarray(cat[col], dtype=float)
            return z, col
    return np.full(len(cat), np.nan, dtype=float), "none"


def _resolve_optional_path(path_str, project_root):
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def _get_catalog_ids(cat, catalog_id_col, row_indices):
    if catalog_id_col in {"__row_index__", "euclid_idx", "row_index"}:
        return np.asarray(row_indices)
    if catalog_id_col not in cat.colnames:
        raise ValueError(
            f"Catalog ID column '{catalog_id_col}' not found. "
            f"Use --catalog-id-column __row_index__ to match with euclid_idx row indices."
        )
    return np.asarray(cat[catalog_id_col])


def _crossmatch_photoz(cat, matched_cat, catalog_id_col, matched_id_col, matched_photoz_col):
    for col in [matched_id_col, matched_photoz_col]:
        if col not in matched_cat.colnames:
            raise ValueError(f"Missing required column for matched photo-z: {col}")

    if "__catalog_match_id" not in cat.colnames:
        raise ValueError("Internal error: __catalog_match_id missing before crossmatch")

    cat_ids = np.asarray(cat["__catalog_match_id"])
    matched_ids = np.asarray(matched_cat[matched_id_col])
    matched_z = np.asarray(matched_cat[matched_photoz_col], dtype=float)

    z_out = np.full(len(cat_ids), np.nan, dtype=float)

    valid_matched = np.isfinite(matched_z)
    if not np.any(valid_matched):
        return z_out

    try:
        cat_ids_i = cat_ids.astype(np.int64)
        matched_ids_i = matched_ids.astype(np.int64)

        mids = matched_ids_i[valid_matched]
        mz = matched_z[valid_matched]
        order = np.argsort(mids)
        mids = mids[order]
        mz = mz[order]

        first_idx = np.unique(mids, return_index=True)[1]
        mids_u = mids[first_idx]
        mz_u = mz[first_idx]

        idx = np.searchsorted(mids_u, cat_ids_i)
        hit = (idx < len(mids_u))
        hit[hit] &= (mids_u[idx[hit]] == cat_ids_i[hit])
        z_out[hit] = mz_u[idx[hit]]
        return z_out
    except Exception:
        z_map = {}
        for mid, zz in zip(matched_ids[valid_matched], matched_z[valid_matched]):
            key = str(mid).strip()
            if key not in z_map:
                z_map[key] = zz
        for i, cid in enumerate(cat_ids):
            z_out[i] = z_map.get(str(cid).strip(), np.nan)
        return z_out


def _load_norm_stats(norm_stats_file):
    data = np.load(norm_stats_file, allow_pickle=True)
    theta_mu = np.asarray(data["theta_mu"], dtype=float)
    theta_sigma = np.asarray(data["theta_sigma"], dtype=float)
    theta_sigma = np.where(theta_sigma < 1e-6, 1.0, theta_sigma)
    labels = np.asarray(data.get("labels", np.array([], dtype=object)), dtype=object)
    return theta_mu, theta_sigma, labels


def _norm_label(label):
    return "".join(str(label).lower().split())


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
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--min-detected-filters", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Detection threshold for using flux+err directly; lower values are treated as non-detections")
    parser.add_argument("--sample-with", choices=["rejection", "mcmc"], default="rejection")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--redshift-columns", default="lp_zBEST,photoz,zbest,z_spec,zphot,ez_z_phot")
    parser.add_argument("--matched-file", default="obs/obs_properties/matched_euclid_farmer.fits",
                        help="Matched Euclid-FARMER FITS for photo-z crossmatch (set empty to disable)")
    parser.add_argument("--catalog-id-column", default="__row_index__",
                        help="ID column in COSMOS_DEEP catalog to match with matched-file; use __row_index__ for euclid_idx")
    parser.add_argument("--matched-euclid-column", default="euclid_idx",
                        help="Euclid index column in matched-file")
    parser.add_argument("--matched-photoz-column", default="lp_zbest",
                        help="Photo-z column in matched-file")
    parser.add_argument("--norm-stats-file", default=None,
                        help="Optional normalization stats .npz; default auto-detect from model-name")
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

    cat_all = Table.read(args.fits_file)
    row_index_all = np.arange(len(cat_all), dtype=np.int64)
    cat = _select_patch(cat_all, args.patch_id)
    # Recompute mask robustly without changing original selection semantics
    patch_col = cat_all["patch_id_list"]
    try:
        mask_int = patch_col == int(args.patch_id) if args.patch_id is not None else np.ones(len(cat_all), dtype=bool)
    except (TypeError, ValueError):
        mask_int = np.zeros(len(cat_all), dtype=bool)
    if args.patch_id is None:
        mask = np.ones(len(cat_all), dtype=bool)
    else:
        mask_str = np.array([str(v).strip() == str(args.patch_id) for v in patch_col])
        mask = mask_int | mask_str
    row_index_patch = row_index_all[mask]

    cat_ids_patch = _get_catalog_ids(cat, args.catalog_id_column, row_index_patch)
    cat["__catalog_match_id"] = cat_ids_patch

    if len(cat) == 0:
        raise ValueError(f"No rows found for patch_id={args.patch_id}")

    z_all = np.full(len(cat), np.nan, dtype=float)
    z_source_parts = []

    matched_file = args.matched_file.strip() if isinstance(args.matched_file, str) else args.matched_file
    if matched_file:
        matched_path = _resolve_optional_path(matched_file, project_root)
        if matched_path is not None and matched_path.exists():
            matched_cat = Table.read(matched_path)
            z_matched = _crossmatch_photoz(
                cat,
                matched_cat,
                catalog_id_col=args.catalog_id_column,
                matched_id_col=args.matched_euclid_column,
                matched_photoz_col=args.matched_photoz_column,
            )
            has_match = np.isfinite(z_matched)
            z_all[has_match] = z_matched[has_match]
            z_source_parts.append(
                f"{args.matched_photoz_column} from {matched_path.name} via {args.catalog_id_column}->{args.matched_euclid_column} ({np.sum(has_match)}/{len(cat)})"
            )
        else:
            print(f"WARNING: matched-file not found: {matched_path}. Falling back to in-catalog redshift columns.")

    missing_z = ~np.isfinite(z_all)
    if np.any(missing_z):
        z_catalog, catalog_source = _choose_redshift(cat, _parse_redshift_columns(args.redshift_columns))
        fill = missing_z & np.isfinite(z_catalog)
        if np.any(fill):
            z_all[fill] = z_catalog[fill]
            z_source_parts.append(f"{catalog_source} in-catalog fallback ({np.sum(fill)} rows)")

    z_source = " + ".join(z_source_parts) if z_source_parts else "unknown"

    n_obj = len(cat) if args.n_max <= 0 else min(args.n_max, len(cat))
    cat = cat[:n_obj]
    z_all = z_all[:n_obj]
    catalog_ids_all = np.asarray(cat["__catalog_match_id"])[:n_obj]

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
    keep_det = n_detected_filters >= args.min_detected_filters
    keep_z = np.isfinite(z_all)
    keep = keep_det & keep_z
    if not np.any(keep):
        raise ValueError(
            f"No galaxies satisfy both: >= {args.min_detected_filters} detected filters and finite photo-z"
        )

    phot_arr = phot_arr[keep]
    sigma_arr = sigma_arr[keep]
    z_use = z_all[keep]
    n_detected_filters = n_detected_filters[keep]
    catalog_ids_use = catalog_ids_all[keep]

    print(f"Catalog rows in patch {args.patch_id}: {n_obj}")
    print(f"Kept for inference: {len(phot_arr)} (min detected filters = {args.min_detected_filters}, finite photo-z required)")
    print(f"Dropped due to missing photo-z: {np.sum(~keep_z)}")
    print(f"Redshift source: {z_source}")
    print("Detected-filter percentiles (10/50/90): "
          f"{np.percentile(n_detected_filters, [10, 50, 90]).astype(int)}")

    posterior_kwargs = {}
    if args.sample_with != "rejection":
        posterior_kwargs["sample_with"] = args.sample_with

    posterior_result = sx.get_posteriors_resolved(
        np.copy(phot_arr),
        args.patch_id if args.patch_id is not None else 0,
        input_z=z_use,
        n_samples=args.n_samples,
        save=False,
        return_stats=False,
        sigma_arr=np.copy(sigma_arr),
        device=args.device,
        **posterior_kwargs,
    )
    posteriors = posterior_result[0] if isinstance(posterior_result, tuple) else posterior_result

    posterior_median = np.median(posteriors, axis=1)
    posterior_std = np.std(posteriors, axis=1)

    norm_stats_file = args.norm_stats_file
    if norm_stats_file is None:
        model_stem = Path(args.model_name).stem
        norm_stats_file = str(project_root / "library" / f"norm_stats_{model_stem}.npz")

    norm_path = Path(norm_stats_file)
    if norm_path.exists():
        theta_mu, theta_sigma, stats_labels = _load_norm_stats(norm_path)
        model_labels = np.array(sx.labels[:posteriors.shape[-1]], dtype=object)
        stats_map = {}
        for i, lab in enumerate(stats_labels):
            if i < len(theta_mu) and i < len(theta_sigma):
                stats_map[_norm_label(lab)] = (float(theta_mu[i]), float(theta_sigma[i]))

        applied = []
        missing = []
        for j, lab in enumerate(model_labels):
            key = _norm_label(lab)
            if key in stats_map:
                mu_j, sig_j = stats_map[key]
                posteriors[:, :, j] = posteriors[:, :, j] * sig_j + mu_j
                applied.append(str(lab))
            else:
                missing.append(str(lab))

        posterior_median = np.median(posteriors, axis=1)
        posterior_std = np.std(posteriors, axis=1)
        print(f"Applied de-normalization from: {norm_path}")
        print(f"Applied norm stats for labels: {applied}")
        if missing:
            print(f"WARNING: no matching norm stats labels for: {missing}")
    else:
        print(f"WARNING: normalization stats file not found: {norm_path}")

    labels_now = np.array(sx.labels[:posteriors.shape[-1]], dtype=object)
    sfr_idx = next((i for i, lab in enumerate(labels_now) if "sfr" in str(lab).lower()), None)
    if sfr_idx is not None:
        print(
            f"SBI SFR min/max: {posterior_median[:, sfr_idx].min():.4f}, "
            f"{posterior_median[:, sfr_idx].max():.4f}"
        )

    out_file = logs_dir / "cosmos_posteriors.npz"
    np.savez_compressed(
        out_file,
        posteriors=posteriors,
        posterior_median=posterior_median,
        posterior_std=posterior_std,
        redshift=z_use,
        catalog_id=catalog_ids_use,
        n_detected_filters=n_detected_filters,
        labels=np.array(sx.labels[:posteriors.shape[-1]], dtype=object),
        model_name=np.array([args.model_name], dtype=object),
        sample_with=np.array([args.sample_with], dtype=object),
        norm_stats_file=np.array([str(norm_path)], dtype=object),
    )
    print(f"Saved: {out_file}")
    print("Saved outputs are in physical space when normalization stats are available.")


if __name__ == "__main__":
    main()
