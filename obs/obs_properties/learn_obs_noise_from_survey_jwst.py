"""Generate observational noise model from COSMOS-Web master catalog JWST photometry."""

import numpy as np
from astropy.table import Table
import os

# NumPy <2.0 uses trapz; >=2.0 uses trapezoid
def _trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return getattr(np, "trapz")(y, x)

FITS_PATH = "/mnt/data_proj/iac18_aasensio_shared/sbi_euclid/SBIEuclid/obs/obs_properties/COSMOS/COSMOSWeb_mastercatalog_v1.fits"
FILTER_LIST_FILE = "filters_to_use_jwst.dat"
FILTER_DIR = "."
OUT_DIR = "."

PERCENTILE_CUTS = [5.0, 15.0, 30.0, 50.0, 70.0, 90.0]
SNR_THRESHOLD = 3.0

def load_filter_metadata(filter_list_file, filt_dir):
    """Load filter metadata from .dat file."""
    entries = []
    with open(os.path.join(filt_dir, filter_list_file)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected 3 columns, got {len(parts)}: {line!r}")
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

def load_jwst_photometry(fits_path, entries):
    """Load JWST aperture photometry from master catalog HDU1."""
    cat = Table.read(fits_path, hdu=1)
    print(f"  Total rows in HDU1: {len(cat)}")

    phot_list = []
    err_list = []

    for entry in entries:
        col_stem = entry["col_stem"]
        # For JWST aperture photometry: flux_aper_f{band} / flux_err_aper_f{band}
        # Note: these columns are 2D arrays (n_galaxies, n_apertures); extract first aperture
        fcol = f"flux_aper_{col_stem}"
        ecol = f"flux_err_aper_{col_stem}"

        if fcol not in cat.colnames or ecol not in cat.colnames:
            raise KeyError(f"Columns '{fcol}' or '{ecol}' not found. "
                         f"Available flux cols: {[c for c in cat.colnames if 'flux' in c][:10]}")

        # Extract first measurement (aperture 0) for each galaxy
        flux_2d = np.array(cat[fcol], dtype=float)
        err_2d = np.array(cat[ecol], dtype=float)
        if flux_2d.ndim == 2:
            flux_scalar = flux_2d[:, 0]
            err_scalar = err_2d[:, 0]
        else:
            flux_scalar = flux_2d
            err_scalar = err_2d

        phot_list.append(flux_scalar)
        err_list.append(err_scalar)

    phot = np.vstack(phot_list)
    err = np.vstack(err_list)
    print(f"  Photometry shape: {phot.shape}")
    return phot, err

def compute_noise_features(phot_ujy, err_ujy, percentile_cuts, snr_threshold=3.0):
    """Compute magnitude bins and uncertainty statistics."""
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

    for i in range(n_filters):
        mags = mag[i, :]
        sigs = mag_err[i, :]
        finite = np.isfinite(mags)

        edges = np.nanpercentile(mags[finite], percentile_cuts)
        percentiles[:, i] = edges

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

def compute_background_limits(phot_ujy, err_ujy, snr_threshold=3.0, faint_percentile=20.0):
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
    print("=" * 70)
    print("JWST NOISE FEATURE COMPUTATION (COSMOS-Web Master Catalog)")
    print("=" * 70)

    print("\n1. Loading filter metadata...")
    entries = load_filter_metadata(FILTER_LIST_FILE, FILTER_DIR)
    n_filters = len(entries)
    print(f"   {n_filters} filters: {', '.join(e['short'] for e in entries)}")

    print("\n2. Computing effective wavelengths...")
    lam_eff = compute_lambda_eff(entries)
    print(f"   lam_eff shape: {lam_eff.shape}")
    print(f"   SNR threshold: {SNR_THRESHOLD}")

    print(f"\n3. Loading JWST photometry from {os.path.basename(FITS_PATH)} HDU1...")
    phot, err = load_jwst_photometry(FITS_PATH, entries)

    print(f"\n4. Computing noise features...")
    percentiles, mean_sigma, std_sigma, sigma_samples = compute_noise_features(
        phot, err, PERCENTILE_CUTS, SNR_THRESHOLD
    )
    limits = compute_background_limits(phot, err, SNR_THRESHOLD)

    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = "cweb_jwst"

    print(f"\n5. Saving output files (prefix: {prefix})...")
    np.save(os.path.join(OUT_DIR, f"lam_eff_{prefix}.npy"), lam_eff)
    print(f"   ✓ lam_eff_{prefix}.npy")

    np.save(os.path.join(OUT_DIR, f"percentiles_{prefix}.npy"), percentiles)
    print(f"   ✓ percentiles_{prefix}.npy")

    np.save(os.path.join(OUT_DIR, f"mean_sigma_{prefix}.npy"), mean_sigma)
    print(f"   ✓ mean_sigma_{prefix}.npy")

    np.save(os.path.join(OUT_DIR, f"std_sigma_{prefix}.npy"), std_sigma)
    print(f"   ✓ std_sigma_{prefix}.npy")

    np.save(os.path.join(OUT_DIR, f"sigma_samples_{prefix}.npy"), sigma_samples, allow_pickle=True)
    print(f"   ✓ sigma_samples_{prefix}.npy")

    np.save(os.path.join(OUT_DIR, f"background_noise_{prefix}.npy"), limits)
    print(f"   ✓ background_noise_{prefix}.npy")

    print(f"\nDone! All files saved to {os.path.abspath(OUT_DIR)}/")

if __name__ == "__main__":
    main()
