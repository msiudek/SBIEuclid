"""Generate observational feature files from COSMOS-Deep FITS catalog."""

import numpy as np
from astropy.table import Table
import os

# NumPy <2.0 uses trapz; >=2.0 uses trapezoid
def _trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return getattr(np, "trapz")(y, x)

# Hard-coded configuration
FITS_PATH = "COSMOS_DEEP.fits"
FILTER_LIST_FILE = "filters_to_use.dat"
FILTER_DIR = "."
OUT_DIR = "."
# Photometry types to process:
#   '2fwhm', '3fwhm' → aperture photometry  (flux_{stem}_{type}_aper)
#   'templfit'       → template-fit (flux_{stem}_templfit; VIS uses flux_vis_psf)
#   'total'          → total flux:  F_band = flux_detection_total × (F_band_2fwhm / F_vis_2fwhm)
#                      Uses aperture color ratios to distribute the per-galaxy total flux
#                      across bands, consistent with Euclid pipeline convention.
PHOT_TYPES = ["total"]

# Detection band used as the reference for 'total' photometry
DETECTION_TOTAL_COL = "flux_detection_total"
DETECTION_APER_STEM = "vis"   # denominator: flux_vis_2fwhm_aper
HEMISPHERE = "north"

PERCENTILE_CUTS = [5.0, 15.0, 30.0, 50.0, 70.0, 90.0]
PATCH_ID = 98
SNR_THRESHOLD = 3.0

def build_phot_col(stem, phot_type, err=False):
    """Return the FITS column name for a given filter stem and photometry type.

    Parameters
    ----------
    stem : str
        Filter col_stem as listed in filters_to_use.dat (e.g. 'h', 'vis', 'g_ext_hsc').
    phot_type : str
        One of '2fwhm', '3fwhm', or 'templfit'.
    err : bool
        If True return the error column, else the flux column.
    """
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        if stem == "vis":
            return f"{prefix}_vis_psf"
        return f"{prefix}_{stem}_templfit"
    # aperture photometry
    return f"{prefix}_{stem}_{phot_type}_aper"


def load_filter_metadata(filter_list_file, filt_dir):
    """Load filter metadata from .dat file. Returns list of dicts with keys:"""
    entries = []
    with open(os.path.join(filt_dir, filter_list_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{filter_list_file}: expected 3 columns (path short col_stem), "
                    f"got {len(parts)} in: {line!r}"
                )
            rel_path, short, col_stem = parts
            entries.append({
                "path": os.path.join(filt_dir, rel_path),
                "rel_path": rel_path,
                "short": short,
                "col_stem": col_stem,
            })
    return entries


def compute_lambda_eff(entries):
    """Compute effective wavelength per filter (Angstrom)."""
    lam_eff = []
    for entry in entries:
        data = np.loadtxt(entry["path"])
        wave = data[:, 0]
        trans = data[:, 1]
        valid = np.isfinite(wave) & np.isfinite(trans) & (trans > 0)
        w = wave[valid]
        t = trans[valid]
        lam_eff.append(_trapezoid(w * t, w) / _trapezoid(t, w))
    return np.array(lam_eff)


def load_from_fits(fits_path, entries, phot_type, patch_id=98):
    """Load photometry and errors from COSMOS-Deep FITS for one photometry type."""
    cat = Table.read(fits_path)
    print(f"Total rows: {len(cat)}")

    # Filter by patch_id
    patch_col = cat["patch_id_list"]
    patch_int = int(patch_id) if not isinstance(patch_id, (int, np.integer)) else patch_id
    mask_list = []
    for val in patch_col:
        try:
            scalar = np.asarray(val).item()
            mask_list.append(int(scalar) == patch_int)
        except (ValueError, TypeError):
            mask_list.append(False)
    mask = np.array(mask_list, dtype=bool)
    cat = cat[mask]
    print(f"Rows after patch filter: {len(cat)}")

    phot_list = []
    err_list = []

    if phot_type == "total":
        # F_band_total = flux_detection_total × (flux_{stem}_2fwhm_aper / flux_vis_2fwhm_aper)
        # σ_band_total = flux_detection_total × (fluxerr_{stem}_2fwhm_aper / flux_vis_2fwhm_aper)
        # Preserves aperture colors; normalises to total flux in detection band.
        det_total = np.array(cat[DETECTION_TOTAL_COL], dtype=float)
        vis_aper = np.array(cat[f"flux_{DETECTION_APER_STEM}_2fwhm_aper"], dtype=float)
        # Scale factor per galaxy; NaN where vis aperture is non-positive
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(
                np.isfinite(vis_aper) & (vis_aper > 0) & np.isfinite(det_total),
                det_total / vis_aper,
                np.nan,
            )
        print(f"   total-flux scale (det_total/vis_aper): "
              f"p10={np.nanpercentile(scale, 10):.3f} "
              f"p50={np.nanpercentile(scale, 50):.3f} "
              f"p90={np.nanpercentile(scale, 90):.3f}")
        for entry in entries:
            stem = entry["col_stem"]
            aper_col = f"flux_{stem}_2fwhm_aper"
            aper_err_col = f"fluxerr_{stem}_2fwhm_aper"
            if aper_col not in cat.colnames:
                raise KeyError(f"Column '{aper_col}' not found in {fits_path}.")
            band_aper = np.array(cat[aper_col], dtype=float)
            band_err  = np.array(cat[aper_err_col], dtype=float)
            phot_list.append(scale * band_aper)
            err_list.append(scale * band_err)
    else:
        for entry in entries:
            stem = entry["col_stem"]
            fcol = build_phot_col(stem, phot_type, err=False)
            ecol = build_phot_col(stem, phot_type, err=True)
            if fcol not in cat.colnames:
                raise KeyError(f"Column '{fcol}' not found in {fits_path}. "
                               f"Available flux cols sample: {[c for c in cat.colnames if 'flux' in c][:8]}")
            phot_list.append(np.array(cat[fcol], dtype=float))
            err_list.append(np.array(cat[ecol], dtype=float))

    phot = np.vstack(phot_list)
    err = np.vstack(err_list)
    print(f"Photometry shape: {phot.shape}")
    return phot, err


def compute_noise_features(phot_ujy, err_ujy, percentile_cuts, snr_threshold=2.0):
    """
    Compute magnitude bins and uncertainty statistics.

    Returns:
    - percentiles: magnitude bin edges (n_cuts × n_filters)
    - mean_sigma: mean mag error per bin (n_filters × n_bins)
    - std_sigma: std of mag error per bin (n_filters × n_bins)
    - sigma_samples: raw mag error samples per bin (n_filters × n_bins, dtype=object)
    """
    # Only include well-detected galaxies (SNR >= threshold) in the LUT.
    # The training p50~2.66 for VIS comes from OOD atlas galaxies (very faint,
    # sigma from background-noise formula), not from the LUT itself.
    valid = np.isfinite(phot_ujy) & np.isfinite(err_ujy) & (phot_ujy > 0) & (err_ujy > 0)
    snr = np.full_like(phot_ujy, np.nan, dtype=float)
    np.divide(phot_ujy, err_ujy, out=snr, where=valid)
    valid &= np.isfinite(snr) & (snr >= snr_threshold)
    
    # Convert to magnitude: m = -2.5 * log10(flux_ujy / 3631 Jy)
    mag = np.full_like(phot_ujy, np.nan)
    mag_err = np.full_like(phot_ujy, np.nan)
    
    for i in range(phot_ujy.shape[0]):
        for j in range(phot_ujy.shape[1]):
            if valid[i, j]:
                mag[i, j] = -2.5 * np.log10(phot_ujy[i, j] * 1e-6 / 3631.0)
                mag_err[i, j] = 2.5 / np.log(10) * err_ujy[i, j] / phot_ujy[i, j]

    n_filters = phot_ujy.shape[0]
    n_bins = len(percentile_cuts) + 1

    percentiles = np.zeros((len(percentile_cuts), n_filters))
    mean_sigma = np.zeros((n_filters, n_bins))
    std_sigma = np.zeros((n_filters, n_bins))
    sigma_samples = np.empty((n_filters, n_bins), dtype=object)

    # Compute per-filter statistics
    for i in range(n_filters):
        mags = mag[i, :]
        sigs = mag_err[i, :]
        finite = np.isfinite(mags)
        
        # Get magnitude bin edges
        edges = np.nanpercentile(mags[finite], percentile_cuts)
        percentiles[:, i] = edges

        # Build bins: [<edges[0]], [edges[0]-edges[1]], ..., [>=edges[-1]]
        bin_masks = [mags < edges[0]]
        for k in range(len(edges) - 1):
            bin_masks.append((mags >= edges[k]) & (mags < edges[k + 1]))
        bin_masks.append(mags >= edges[-1])

        for j, bin_mask in enumerate(bin_masks):
            values = sigs[bin_mask]
            values_finite = values[np.isfinite(values)]
            sigma_samples[i, j] = values_finite
            
            if len(values_finite) > 0:
                mean_sigma[i, j] = np.mean(values_finite)
                std_sigma[i, j] = np.std(values_finite)

    return percentiles, mean_sigma, std_sigma, sigma_samples


def compute_background_limits(phot_ujy, err_ujy, snr_threshold=3.0, faint_percentile=20.0,
                              depth5_ujy=None):
    """Return the 1-sigma flux limit per filter.

    Uses the median flux error of near-threshold (SNR < snr_threshold) objects.
    This conservative estimate accounts for the effective noise floor of the survey
    and is the appropriate floor for the OOD sigma_mag formula used during training.
    """
    valid = np.isfinite(phot_ujy) & np.isfinite(err_ujy) & (phot_ujy > 0) & (err_ujy > 0)
    snr = np.full_like(phot_ujy, np.nan, dtype=float)
    np.divide(phot_ujy, err_ujy, out=snr, where=valid)

    limits = np.zeros(phot_ujy.shape[0])
    for i in range(phot_ujy.shape[0]):
        err_i = err_ujy[i, valid[i, :]]
        snr_i = snr[i, valid[i, :]]
        low_snr_mask = snr_i < snr_threshold
        if np.any(low_snr_mask):
            limits[i] = np.median(err_i[low_snr_mask])
        elif len(err_i) > 0:
            limits[i] = np.median(err_i)

    return limits


def main():
    print("=" * 60)
    print("NOISE FEATURE COMPUTATION")
    print("=" * 60)

    # Load filter metadata (path, short name, FITS col_stem) from .dat file
    print("\n1. Loading filter metadata...")
    entries = load_filter_metadata(FILTER_LIST_FILE, FILTER_DIR)
    n_filters = len(entries)
    print(f"   {n_filters} filters: {', '.join(e['short'] for e in entries)}")

    # Compute effective wavelengths
    print("\n2. Computing effective wavelengths...")
    lam_eff = compute_lambda_eff(entries)
    print(f"   lam_eff shape: {lam_eff.shape}")
    print(f"   SNR threshold for detections: {SNR_THRESHOLD}")

    # Save outputs directory
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n3. Output directory: {os.path.abspath(OUT_DIR)}")

    # Process each photometry type independently and write separate files
    for phot_type in PHOT_TYPES:
        prefix = f"{HEMISPHERE}_{phot_type}"
        print(f"\n4. Processing phot_type: {phot_type}")
        phot, err = load_from_fits(FITS_PATH, entries, phot_type, PATCH_ID)

        percentiles, mean_sigma, std_sigma, sigma_samples = compute_noise_features(
            phot, err, PERCENTILE_CUTS, SNR_THRESHOLD
        )
        # Load 5σ depths if the depth file exists alongside this script,
        # otherwise fall back to the empirical estimator.
        _depth_file = os.path.join(os.path.dirname(__file__), 'noise_5sighmadepth.py')
        _depth5_ujy = None
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location('_depths', _depth_file)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _order = _mod.order
            _depth5_ujy = np.array([_mod.mag_to_flux_ujy(_mod.depth5_mag[k]) for k in _order],
                                   dtype=float)
            print(f'   Using 5σ depths from noise_5sighmadepth.py for limits.')
        except Exception as _e:
            print(f'   WARNING: could not load 5σ depths ({_e}); using empirical fallback.')
        limits = compute_background_limits(phot, err, SNR_THRESHOLD, depth5_ujy=_depth5_ujy)

        np.save(os.path.join(OUT_DIR, f"lam_eff_{prefix}.npy"), lam_eff)
        print(f"   ✓ lam_eff_{prefix}.npy")
        np.save(os.path.join(OUT_DIR, f"percentiles_{prefix}.npy"), percentiles)
        print(f"   ✓ percentiles_{prefix}.npy")
        np.save(os.path.join(OUT_DIR, f"mean_sigma_{prefix}.npy"), mean_sigma)
        print(f"   ✓ mean_sigma_{prefix}.npy")
        np.save(os.path.join(OUT_DIR, f"std_sigma_{prefix}.npy"), std_sigma)
        print(f"   ✓ std_sigma_{prefix}.npy")
        np.save(os.path.join(OUT_DIR, f"sigma_samples_{prefix}.npy"), sigma_samples)
        print(f"   ✓ sigma_samples_{prefix}.npy")
        # background_noise is calibrated from the committed v5.1 values — do not regenerate.
        # Run:  git restore obs/obs_properties/background_noise_{prefix}.npy
        # if it was accidentally overwritten.
        print(f"   (skipping background_noise_{prefix}.npy — use committed git version)")
        print(f"   Empirical sigma_lim (uJy): {np.array2string(limits, precision=5)}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()