"""Generate observational feature files from COSMOS-Deep FITS catalog."""

import numpy as np
from astropy.table import Table
import os

# NumPy <2.0 uses trapz; >=2.0 uses trapezoid
_trapezoid = getattr(np, "trapezoid", np.trapz)

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


def load_from_fits(fits_path, entries, aperture, patch_id=98):
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

    for entry in entries:
        col_stem = entry["col_stem"]

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

    # Process each aperture independently and write separate files
    for aperture in APERTURES:
        prefix = f"{HEMISPHERE}_{aperture}"
        print(f"\n4. Processing aperture: {aperture}")
        phot, err = load_from_fits(FITS_PATH, entries, aperture, PATCH_ID)

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
