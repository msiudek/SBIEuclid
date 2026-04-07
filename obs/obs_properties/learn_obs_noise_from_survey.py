"""Generate observational feature files from COSMOS-Deep FITS catalog."""

import numpy as np
from astropy.table import Table
import os

# Hard-coded configuration
FITS_PATH = "COSMOS_DEEP.fits"
FILTER_LIST_FILE = "filters_to_use.dat"
FILTER_DIR = "."
OUT_DIR = "."
APERTURES = ["2fwhm", "3fwhm"]
HEMISPHERE = "north"

PERCENTILE_CUTS = [5.0, 15.0, 30.0, 50.0, 70.0, 90.0]
PATCH_ID = 98
SNR_THRESHOLD = 2.0

FILTER_STEM_TO_COL = {
    "CFHT_MegaCam.r":    "r_ext_megacam",
    "CFHT_MegaCam.u":    "u_ext_megacam",
    "Euclid_NISP.H":     "h",
    "Euclid_NISP.J":     "j",
    "Euclid_NISP.Y":     "y",
    "Euclid_VIS.vis":    "vis",
    "Subaru_HSC.g":      "g_ext_hsc",
    "Subaru_HSC.z":      "z_ext_hsc",
    "PAN-STARRS_PS1.i":  "i_ext_panstarrs",
    "CTIO_DECam.g":      "g_ext_decam",
    "CTIO_DECam.r":      "r_ext_decam",
    "CTIO_DECam.i":      "i_ext_decam",
    "CTIO_DECam.z":      "z_ext_decam",
}


def load_filter_paths(filter_list_file, filt_dir):
    """Load filter file paths from list."""
    with open(os.path.join(filt_dir, filter_list_file)) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [os.path.join(filt_dir, line) for line in lines]


def compute_lambda_eff(filter_paths):
    """Compute effective wavelength per filter (Angstrom)."""
    lam_eff = []
    for path in filter_paths:
        data = np.loadtxt(path)
        wave = data[:, 0]
        trans = data[:, 1]
        valid = np.isfinite(wave) & np.isfinite(trans) & (trans > 0)
        w = wave[valid]
        t = trans[valid]
        lam_eff.append(np.trapezoid(w * t, w) / np.trapezoid(t, w))
    return np.array(lam_eff)


def extract_filter_stem(filter_path):
    """'FILTERS_CFHT/CFHT_MegaCam.r.dat' → 'CFHT_MegaCam.r'"""
    basename = os.path.basename(filter_path)
    return os.path.splitext(basename)[0]


def load_from_fits(fits_path, filter_rel_paths, aperture, patch_id=98):
    """Load photometry and errors from COSMOS-Deep FITS for one aperture."""
    cat = Table.read(fits_path)
    print(f"Total rows: {len(cat)}")
    
    # Filter by patch_id (convert to int first for comparison)
    patch_col = cat["patch_id_list"]
    patch_int = int(patch_id) if not isinstance(patch_id, (int, np.integer)) else patch_id
    # Build mask by direct iteration to avoid numpy scalar conversion warnings
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

    for rel_path in filter_rel_paths:
        stem = extract_filter_stem(rel_path)
        col_stem = FILTER_STEM_TO_COL[stem]

        fcol = f"flux_{col_stem}_{aperture}_aper"
        ecol = f"fluxerr_{col_stem}_{aperture}_aper"
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
    # Detection regime only: finite, positive, and SNR above threshold
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


def compute_background_limits(phot_ujy, err_ujy, snr_threshold=2.0, faint_percentile=20.0):
    """Estimate 1-sigma background limit from non-detection (low-SNR) regime."""
    valid = np.isfinite(phot_ujy) & np.isfinite(err_ujy) & (phot_ujy > 0) & (err_ujy > 0)
    snr = np.full_like(phot_ujy, np.nan, dtype=float)
    np.divide(phot_ujy, err_ujy, out=snr, where=valid)
    
    limits = np.zeros(phot_ujy.shape[0])
    for i in range(phot_ujy.shape[0]):
        phot_i = phot_ujy[i, valid[i, :]]
        err_i = err_ujy[i, valid[i, :]]
        snr_i = snr[i, valid[i, :]]

        if len(phot_i) > 0:
            low_snr_mask = np.isfinite(snr_i) & (snr_i < snr_threshold)
            if np.any(low_snr_mask):
                limits[i] = np.median(err_i[low_snr_mask])
            else:
                cut = np.percentile(phot_i, faint_percentile)
                faint_mask = phot_i <= cut
                limits[i] = np.median(err_i[faint_mask])

    return limits


def main():
    print("=" * 60)
    print("NOISE FEATURE COMPUTATION")
    print("=" * 60)

    # Load filter transmission curves
    print("\n1. Loading filter curves...")
    filter_paths = load_filter_paths(FILTER_LIST_FILE, FILTER_DIR)
    n_filters = len(filter_paths)
    print(f"   {n_filters} filters loaded")

    # Compute effective wavelengths
    print("\n2. Computing effective wavelengths...")
    lam_eff = compute_lambda_eff(filter_paths)
    print(f"   lam_eff shape: {lam_eff.shape}")
    print(f"   SNR threshold for detections: {SNR_THRESHOLD}")

    # Save outputs directory
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n3. Output directory: {os.path.abspath(OUT_DIR)}")

    # Build ordered filter list entries
    filter_rel_paths = [line.strip() for line in open(FILTER_LIST_FILE)
                        if line.strip() and not line.startswith("#")]

    # Process each aperture independently and write separate files
    for aperture in APERTURES:
        prefix = f"{HEMISPHERE}_{aperture}"
        print(f"\n4. Processing aperture: {aperture}")
        phot, err = load_from_fits(FITS_PATH, filter_rel_paths, aperture, PATCH_ID)

        percentiles, mean_sigma, std_sigma, sigma_samples = compute_noise_features(
            phot, err, PERCENTILE_CUTS, SNR_THRESHOLD
        )
        limits = compute_background_limits(phot, err, SNR_THRESHOLD)

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
        np.save(os.path.join(OUT_DIR, f"background_noise_{prefix}.npy"), limits)
        print(f"   ✓ background_noise_{prefix}.npy")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
