"""Generate observational feature files for SBIPIX from a survey catalog.

This script supports two input modes:

1. Pre-computed .npy arrays (--phot-file / --err-file):
   Pass photometry and uncertainty arrays in microJy directly.

2. COSMOS-Deep FITS catalog (--fits-file):
   Reads the Euclid COSMOS-Deep FITS catalog, selects a specific patch
   (--patch-id, default 98), and pools both 2fwhm and 3fwhm aperture
   measurements to build robust noise statistics.  Filter → column mapping
   is resolved automatically from the filter filenames in the filter list.

Outputs (with --prefix <prefix>):
  lam_eff_<prefix>.npy           effective wavelength per filter (Angstrom)
  percentiles_<prefix>.npy       magnitude-bin boundaries (n_cuts × n_filters)
  mean_sigma_<prefix>.npy        mean mag uncertainty per bin  (n_filters × n_bins)
  std_sigma_<prefix>.npy         std  mag uncertainty per bin  (n_filters × n_bins)
  background_noise_<prefix>.npy  1-sigma flux limit per filter (microJy)
"""

import argparse
import os
import warnings
import numpy as np

# ---------------------------------------------------------------------------
# Mapping from filter filename stem (basename without extension) to the flux
# column stem used in the COSMOS-Deep FITS table.
# Full column names are:  flux_<stem>_2fwhm_aper  /  flux_<stem>_3fwhm_aper
# ---------------------------------------------------------------------------
FILTER_STEM_TO_COL = {
    # CFHT MegaCam
    "CFHT_MegaCam.r":    "r_ext_megacam",
    "CFHT_MegaCam.u":    "u_ext_megacam",
    # Euclid NISP
    "Euclid_NISP.H":     "h",
    "Euclid_NISP.J":     "j",
    "Euclid_NISP.Y":     "y",
    # Euclid VIS
    "Euclid_VIS.vis":    "vis",
    # Subaru HSC
    "Subaru_HSC.g":      "g_ext_hsc",
    "Subaru_HSC.z":      "z_ext_hsc",
    # Pan-STARRS PS1
    "PAN-STARRS_PS1.i":  "i_ext_panstarrs",
    # DECam (kept for completeness)
    "CTIO_DECam.g":      "g_ext_decam",
    "CTIO_DECam.r":      "r_ext_decam",
    "CTIO_DECam.i":      "i_ext_decam",
    "CTIO_DECam.z":      "z_ext_decam",
}


def load_filter_paths(filter_list_file, filt_dir):
    """Return ordered list of filter transmission file paths from filter list."""
    filter_list_path = filter_list_file
    if not os.path.isabs(filter_list_path):
        filter_list_path = os.path.join(filt_dir, filter_list_file)

    with open(filter_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    rel_paths = [line for line in lines if line and not line.startswith("#")]
    return [os.path.join(filt_dir, rel_path) for rel_path in rel_paths]


def compute_lambda_eff_from_curves(filter_paths):
    """Compute effective wavelength for each transmission curve (Angstrom)."""
    lam_eff = []
    for path in filter_paths:
        data = np.loadtxt(path)
        wave = data[:, 0]
        trans = data[:, 1]
        valid = np.isfinite(wave) & np.isfinite(trans) & (trans > 0)
        if np.any(valid):
            w = wave[valid]
            t = trans[valid]
            denom = np.trapz(t, w)
            if denom > 0:
                lam_eff.append(np.trapz(w * t, w) / denom)
            else:
                lam_eff.append(np.nan)
        else:
            lam_eff.append(np.nan)
    return np.asarray(lam_eff)


def _to_filter_major(arr, n_filters):
    """Ensure array shape is (n_filters, n_samples)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[0] == n_filters:
        return arr
    if arr.shape[1] == n_filters:
        return arr.T
    raise ValueError(
        f"Cannot align array shape {arr.shape} with n_filters={n_filters}. "
        "Expected one axis to equal n_filters."
    )


def compute_noise_features(phot_ujy, err_ujy, percentile_cuts,
                           min_bin_population=20, snr_min=-5.0,
                           mag_err_clip_max=2.0, std_sigma_clip_max=1.0):
    """Compute percentile bins and mag uncertainty statistics per filter."""
    tiny = 1e-30
    valid = np.isfinite(phot_ujy) & np.isfinite(err_ujy) & (err_ujy > 0)
    snr = np.full_like(phot_ujy, np.nan, dtype=float)
    np.divide(phot_ujy, err_ujy, out=snr, where=valid)
    valid &= np.isfinite(snr) & (snr > snr_min)
    valid_mag = valid & (phot_ujy > tiny)

    phot_safe = np.where(valid_mag, phot_ujy, np.nan)
    err_safe = np.where(valid_mag, err_ujy, np.nan)

    mag = -2.5 * np.log10((phot_safe * 1e-6) / 3631.0)
    mag_err = np.abs(-2.5 / np.log(10) * err_safe / phot_safe)
    mag_err = np.clip(mag_err, 0.0, mag_err_clip_max)

    n_filters = phot_ujy.shape[0]
    n_bins = len(percentile_cuts) + 1

    percentiles = np.full((len(percentile_cuts), n_filters), np.nan)
    mean_sigma = np.full((n_filters, n_bins), np.nan)
    std_sigma = np.full((n_filters, n_bins), np.nan)

    for i in range(n_filters):
        mags_i = mag[i, :]
        sig_i = mag_err[i, :]

        finite_mag = np.isfinite(mags_i)
        if np.sum(finite_mag) == 0:
            warnings.warn(
                f"Filter index {i} has no valid magnitude values after cuts; "
                "noise bins will remain NaN.",
                RuntimeWarning,
            )
            continue

        edges = np.nanpercentile(mags_i, percentile_cuts)
        percentiles[:, i] = edges

        masks = []
        masks.append(mags_i < edges[0])
        for k in range(len(edges) - 1):
            masks.append((mags_i >= edges[k]) & (mags_i < edges[k + 1]))
        masks.append(mags_i >= edges[-1])

        for j, mask in enumerate(masks):
            vals = sig_i[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size >= min_bin_population:
                mean_sigma[i, j] = np.nanmean(vals)
                std_sigma[i, j] = np.minimum(np.nanstd(vals), std_sigma_clip_max)

    return percentiles, mean_sigma, std_sigma


def compute_background_limits(phot_ujy, err_ujy, sigma_level=1.0,
                              faint_percentile=20.0, snr_min=-5.0):
    """Estimate per-filter background limits (microJy) from faint-end errors."""
    valid = np.isfinite(phot_ujy) & np.isfinite(err_ujy) & (err_ujy > 0)
    snr = np.full_like(phot_ujy, np.nan, dtype=float)
    np.divide(phot_ujy, err_ujy, out=snr, where=valid)
    valid &= np.isfinite(snr) & (snr > snr_min)

    limits = np.full(phot_ujy.shape[0], np.nan)
    for i in range(phot_ujy.shape[0]):
        phot_i = phot_ujy[i, valid[i, :]]
        err_i = err_ujy[i, valid[i, :]]

        if phot_i.size == 0:
            warnings.warn(
                f"Filter index {i} has no valid samples for background depth; "
                "background limit will be NaN.",
                RuntimeWarning,
            )
            continue

        cut = np.nanpercentile(phot_i, faint_percentile)
        faint_mask = phot_i <= cut
        if np.sum(faint_mask) == 0:
            faint_mask = np.ones_like(phot_i, dtype=bool)

        limits[i] = sigma_level * np.nanmedian(err_i[faint_mask])

    return limits


def _filter_stem(filter_rel_path):
    """Extract the stem key used in FILTER_STEM_TO_COL from a filter list entry.

    E.g. 'FILTERS_CFHT/CFHT_MegaCam.r.dat' → 'CFHT_MegaCam.r'
    """
    basename = os.path.basename(filter_rel_path)          # 'CFHT_MegaCam.r.dat'
    # strip only the .dat / .txt suffix, keep the rest (e.g. 'MegaCam.r')
    root, ext = os.path.splitext(basename)
    if ext.lower() in (".dat", ".txt"):
        return root                                        # 'CFHT_MegaCam.r'
    return basename


def load_from_cosmos_fits(fits_path, filter_rel_paths, patch_id=98,
                          flux_to_ujy=1.0, apertures=("2fwhm", "3fwhm"),
                          warn_min_valid=1000):
    """Load photometry from COSMOS-Deep FITS and return (phot_ujy, err_ujy).

    Parameters
    ----------
    fits_path : str
        Path to the ``cosmos_deep.fits`` catalog.
    filter_rel_paths : list of str
        Relative filter paths from the filter list (same order as filter list).
    patch_id : int or str, default 98
        Value of ``patch_id_list`` to select.  Matched as both string and int.
    flux_to_ujy : float, default 1.0
        Scale factor applied to raw catalog fluxes to convert to microJy.
    apertures : tuple of str, default ('2fwhm', '3fwhm')
        Aperture suffixes to pool.  Each gives column
        ``flux_<stem>_<ap>_aper`` / ``fluxerr_<stem>_<ap>_aper``.
    warn_min_valid : int, default 1000
        Emit warning if pooled valid samples per filter are below this threshold.

    Returns
    -------
    phot_ujy : ndarray, shape (n_filters, n_samples)
    err_ujy  : ndarray, shape (n_filters, n_samples)
    """
    from astropy.table import Table

    print(f"Reading FITS catalog: {fits_path}")
    cat = Table.read(fits_path)
    print(f"  Total rows: {len(cat)}")

    # ------------------------------------------------------------------
    # Filter by patch_id_list == patch_id  (handle string or numeric col)
    # ------------------------------------------------------------------
    patch_col = cat["patch_id_list"]
    try:
        # numeric comparison first
        mask = patch_col == int(patch_id)
    except (ValueError, TypeError):
        mask = np.zeros(len(cat), dtype=bool)

    # Also try string comparison (column may store '98' as a string)
    str_patch = str(patch_id)
    try:
        str_mask = np.array([str(v).strip() == str_patch for v in patch_col])
        mask = mask | str_mask
    except Exception:
        pass

    cat = cat[mask]
    print(f"  Rows after patch_id_list == {patch_id}: {len(cat)}")
    if len(cat) == 0:
        raise ValueError(f"No rows found with patch_id_list == {patch_id}")

    # ------------------------------------------------------------------
    # Build phot / err arrays by pooling apertures
    # ------------------------------------------------------------------
    phot_list = []
    err_list = []

    for rel_path in filter_rel_paths:
        stem = _filter_stem(rel_path)
        col_stem = FILTER_STEM_TO_COL.get(stem)
        if col_stem is None:
            raise KeyError(
                f"No FITS column mapping for filter stem '{stem}'. "
                f"Add it to FILTER_STEM_TO_COL in the script."
            )

        phot_aps, err_aps = [], []
        n_valid_total = 0
        for ap in apertures:
            fcol = f"flux_{col_stem}_{ap}_aper"
            ecol = f"fluxerr_{col_stem}_{ap}_aper"
            if fcol not in cat.colnames:
                raise KeyError(f"Column '{fcol}' not found in FITS table. "
                               f"Available: {cat.colnames[:20]} ...")
            phot_ap = np.array(cat[fcol], dtype=float) * flux_to_ujy
            err_ap = np.array(cat[ecol], dtype=float) * flux_to_ujy

            valid = np.isfinite(phot_ap) & np.isfinite(err_ap) & (phot_ap > 0) & (err_ap > 0)
            n_valid_total += int(np.sum(valid))

            phot_aps.append(phot_ap)
            err_aps.append(err_ap)

        if n_valid_total == 0:
            warnings.warn(
                f"Filter '{stem}' has 0 valid samples in patch {patch_id} "
                f"for apertures {tuple(apertures)}. Output stats may be NaN.",
                RuntimeWarning,
            )
        elif n_valid_total < warn_min_valid:
            warnings.warn(
                f"Filter '{stem}' has low valid sample count ({n_valid_total}) "
                f"in patch {patch_id}.",
                RuntimeWarning,
            )

        phot_list.append(np.concatenate(phot_aps))
        err_list.append(np.concatenate(err_aps))

    phot_ujy = np.vstack(phot_list)   # (n_filters, n_samples * n_apertures)
    err_ujy  = np.vstack(err_list)
    print(f"  Loaded phot array shape: {phot_ujy.shape} "
          f"(pooled {len(apertures)} apertures × {len(cat)} objects)")
    return phot_ujy, err_ujy


def main():
    parser = argparse.ArgumentParser(
        description="Generate SBIPIX observational feature files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes (mutually exclusive):
  --fits-file    Read photometry from COSMOS-Deep FITS catalog
                 (pools 2fwhm + 3fwhm apertures for the selected patch).
  --phot-file / --err-file
                 Read pre-computed .npy arrays (units: microJy).
""")
    parser.add_argument("--filter-list", required=True,
                        help="Filter list filename or absolute path")
    parser.add_argument("--filt-dir", default=".",
                        help="Directory containing filter curves (default: .)")

    # --- FITS input mode ---
    grp_fits = parser.add_argument_group("FITS input mode")
    grp_fits.add_argument("--fits-file",
                          help="Path to cosmos_deep.fits catalog")
    grp_fits.add_argument("--patch-id", default="98",
                          help="patch_id_list value to select (default: 98)")
    grp_fits.add_argument("--apertures", nargs="+", default=["2fwhm", "3fwhm"],
                          help="Aperture suffixes to pool (default: 2fwhm 3fwhm)")
    grp_fits.add_argument("--flux-to-ujy", type=float, default=1.0,
                          help="Scale factor: raw_flux × flux-to-ujy = microJy (default: 1.0)")
    grp_fits.add_argument("--warn-min-valid", type=int, default=1000,
                          help="Warn if pooled valid samples per filter are below this value (default: 1000)")

    # --- .npy input mode ---
    grp_npy = parser.add_argument_group(".npy input mode")
    grp_npy.add_argument("--phot-file",
                         help="Path to photometry array (.npy), units: microJy")
    grp_npy.add_argument("--err-file",
                         help="Path to uncertainty array (.npy), units: microJy")

    # --- output options ---
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for generated .npy files")
    parser.add_argument("--percentiles", nargs="+", type=float,
                        default=[5.0, 15.0, 30.0, 50.0, 70.0, 90.0],
                        help="Percentile cuts for mag bins (default: 5 15 30 50 70 90)")
    parser.add_argument("--min-bin-population", type=int, default=20,
                        help="Minimum valid samples required in a mag bin to compute sigma stats (default: 20)")
    parser.add_argument("--snr-min", type=float, default=-5.0,
                        help="Minimum SNR cut applied to valid samples (default: -5)")
    parser.add_argument("--mag-err-clip-max", type=float, default=2.0,
                        help="Maximum allowed magnitude uncertainty used in bin statistics (default: 2.0 mag)")
    parser.add_argument("--std-sigma-clip-max", type=float, default=1.0,
                        help="Maximum allowed std_sigma per bin (default: 1.0 mag)")
    parser.add_argument("--faint-percentile", type=float, default=20.0,
                        help="Use fluxes below this percentile to estimate background depth (default: 20)")
    parser.add_argument("--sigma-limit", type=float, default=1.0,
                        help="Sigma level for background limit file (default: 1.0)")
    parser.add_argument("--prefix", default="euclid",
                        help="Prefix for output files (default: euclid)")
    args = parser.parse_args()

    # --- validate input mode ---
    use_fits = args.fits_file is not None
    use_npy  = (args.phot_file is not None) or (args.err_file is not None)
    if use_fits and use_npy:
        parser.error("Specify either --fits-file OR --phot-file/--err-file, not both.")
    if not use_fits and not use_npy:
        parser.error("Specify either --fits-file or both --phot-file and --err-file.")
    if use_npy and (args.phot_file is None or args.err_file is None):
        parser.error("--phot-file and --err-file must both be provided together.")

    os.makedirs(args.out_dir, exist_ok=True)

    filter_paths = load_filter_paths(args.filter_list, args.filt_dir)
    n_filters = len(filter_paths)
    print(f"Filters: {n_filters}")

    # --- load photometry ---
    if use_fits:
        # resolve relative filter paths (strip the filt_dir prefix to get rel paths)
        filter_list_path = args.filter_list
        if not os.path.isabs(filter_list_path):
            filter_list_path = os.path.join(args.filt_dir, args.filter_list)
        with open(filter_list_path) as f:
            rel_paths = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        phot, err = load_from_cosmos_fits(
            fits_path=args.fits_file,
            filter_rel_paths=rel_paths,
            patch_id=args.patch_id,
            flux_to_ujy=args.flux_to_ujy,
            apertures=args.apertures,
            warn_min_valid=args.warn_min_valid,
        )
    else:
        phot = np.load(args.phot_file)
        err  = np.load(args.err_file)
        phot = _to_filter_major(phot, n_filters)
        err  = _to_filter_major(err,  n_filters)

    lam_eff = compute_lambda_eff_from_curves(filter_paths)
    percentiles, mean_sigma, std_sigma = compute_noise_features(
        phot,
        err,
        args.percentiles,
        min_bin_population=args.min_bin_population,
        snr_min=args.snr_min,
        mag_err_clip_max=args.mag_err_clip_max,
        std_sigma_clip_max=args.std_sigma_clip_max,
    )
    limits = compute_background_limits(
        phot,
        err,
        sigma_level=args.sigma_limit,
        faint_percentile=args.faint_percentile,
        snr_min=args.snr_min,
    )

    out = args.out_dir
    p   = args.prefix
    np.save(os.path.join(out, f"lam_eff_{p}.npy"),           lam_eff)
    np.save(os.path.join(out, f"percentiles_{p}.npy"),        percentiles)
    np.save(os.path.join(out, f"mean_sigma_{p}.npy"),         mean_sigma)
    np.save(os.path.join(out, f"std_sigma_{p}.npy"),          std_sigma)
    np.save(os.path.join(out, f"background_noise_{p}.npy"),   limits)

    print("\nGenerated files:")
    for name, arr in [
        (f"lam_eff_{p}.npy",         lam_eff),
        (f"percentiles_{p}.npy",     percentiles),
        (f"mean_sigma_{p}.npy",      mean_sigma),
        (f"std_sigma_{p}.npy",       std_sigma),
        (f"background_noise_{p}.npy", limits),
    ]:
        print(f"  {os.path.join(out, name)}  shape={arr.shape}")


if __name__ == "__main__":
    main()



