"""
SED processing utilities for SBIPIX.

This module provides shared utilities for:
- Flux and magnitude conversions
- Filter metadata loading
- Photometry data extraction from FITS catalogs
- Band/filter lookups
"""

import numpy as np
from pathlib import Path
import os


def mag_conversion(x, convert_to='mag'):
    """
    Convert between magnitudes and fluxes.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) to convert
    convert_to : str, optional
        Conversion type: 'mag' for magnitudes, 'flux' for fluxes (default: 'mag')

    Returns
    -------
    float or np.ndarray
        Converted value(s)
        
    Notes
    -----
    Uses AB magnitude system with zero point of 3631 Jy.
    For magnitude: mag = -2.5 * log10(flux_μJy / 3631e6)
    For flux: flux_μJy = 3631e6 * 10^(-mag/2.5)
    """
    if convert_to == 'mag':
        return -2.5 * np.log10(x * 1e-6 / 3631)
    elif convert_to == 'flux':  # Return in microjansky
        return (3631 / 1e-6) * 10 ** (x / (-2.5))
    else:
        raise ValueError("convert_to must be 'mag' or 'flux'")


def compute_surface_density(stellar_mass_map, pixel_scale, D_A):
    """
    Compute stellar mass surface density (M☉/kpc²) from stellar mass per pixel.
    
    Parameters
    ----------
    stellar_mass_map : np.ndarray
        Stellar mass per pixel (M☉/pixel)
    pixel_scale : float
        Pixel scale (arcsec/pixel)
    D_A : float
        Angular diameter distance in kpc/arcsec at redshift z

    Returns
    -------
    np.ndarray
        Stellar mass surface density in M☉/kpc²
    """
    pixel_area_kpc2 = (pixel_scale * D_A) ** 2  # Convert pixel area to kpc²
    mass_surface_density = stellar_mass_map / pixel_area_kpc2  # M☉/kpc²
    return mass_surface_density


def compute_surface_density_with_uncertainty(stellar_mass_samples, pixel_scale, D_A):
    """
    Compute stellar mass surface density with uncertainty propagation.
    
    Parameters
    ----------
    stellar_mass_samples : np.ndarray
        Array of log stellar mass samples from posterior
    pixel_scale : float
        Pixel scale (arcsec/pixel)
    D_A : float
        Angular diameter distance in kpc/arcsec
    
    Returns
    -------
    tuple
        (median, std) of mass surface density in M☉/kpc²
    """
    pixel_area_kpc2 = (pixel_scale * D_A) ** 2
    # Transform all samples from log to linear space
    mass_density_samples = 10**stellar_mass_samples / pixel_area_kpc2
    
    return np.median(mass_density_samples), np.std(mass_density_samples)


def escalon(t, ti):
    """
    Step function for delayed SFH models.
    
    Parameters
    ----------
    t : float or np.ndarray
        Time values
    ti : float
        Start time of star formation
        
    Returns
    -------
    bool or np.ndarray
        True where t > ti, False otherwise
    """
    return t > ti


def tau_delayed_SFR(t, M, tau, t_i):
    """
    τ-delayed star formation rate (SFR) model.
    
    SFR(t) = (M/τ²) * (t-t_i) * exp(-(t-t_i)/τ) for t > t_i, 0 otherwise
    
    Parameters
    ----------
    t : float or np.ndarray
        Cosmic time (Gyr)
    M : float
        Normalization (total stellar mass formed; cancels out in ratios)
    tau : float
        Characteristic timescale (Gyr)
    t_i : float
        Initial time of star formation (Gyr)
    
    Returns
    -------
    float or np.ndarray
        Star formation rate at time t
    """
    if t < t_i:
        return 0.0
    else:
        return M / tau**2 * (t - t_i) * np.exp(-(t - t_i) / tau)


def sfh_delayed_exponential(t, logmassval, tau, ti):
    """
    Delayed exponential SFH model from Simha et al. 2014.
    
    Parameters
    ----------
    t : np.ndarray
        Time bins in Gyr
    logmassval : float
        Log stellar mass in M☉
    tau : float
        Timescale of decrease in Gyr
    ti : float
        Time since SF began in Gyr
        
    Returns
    -------
    tuple
        (sfh, timeax) where sfh is SFR in M☉/Gyr and timeax is time axis in Gyr
    """
    from scipy import integrate
    
    # Normalize to get correct total stellar mass
    integral_result = integrate.quad(
        lambda t: (t-ti) * np.exp(-(t-ti)/tau) * escalon(t, ti),
        np.min(t), np.max(t)
    )
    denom = float(integral_result[0])
    if (not np.isfinite(denom)) or (denom <= 0.0):
        sfh = np.zeros_like(t, dtype=float)
        return sfh, t
    A = 10**logmassval / denom
    
    sfh = A * (t-ti) * np.exp(-(t-ti)/tau) * escalon(t, ti)
    return sfh, t  # Units are M☉/Gyr & Gyr


def convert_to_microjansky(spec, zval, cosmo):
    """
    Convert spectrum to microjansky units.
    
    Parameters
    ----------
    spec : np.ndarray
        Spectrum in L☉/Hz
    zval : float
        Redshift
    cosmo : astropy.cosmology object
        Cosmology for distance calculations
        
    Returns
    -------
    np.ndarray
        Spectrum in microjansky
    """
    # Luminosity distance
    d_L = cosmo.luminosity_distance(zval).to('cm').value
    
    # Convert from L☉/Hz to erg/s/Hz
    L_sun = 3.828e33  # erg/s
    spec_erg = spec * L_sun
    
    # Convert to flux at Earth: F = L / (4π d_L² (1+z)).
    # NOTE: we currently keep the (1+z) dimming term enabled (physics-motivated).
    # For controlled A/B tests against older atlases, you can temporarily remove this factor.
    flux_erg = spec_erg / (4 * np.pi * d_L**2 * (1 + zval))
    
    # Convert to Jansky (1 Jy = 1e-23 erg/s/cm²/Hz)
    flux_jy = flux_erg / 1e-23
    
    # Convert to microjansky
    flux_ujy = flux_jy * 1e6
    
    return flux_ujy


# ============================================================================
# Validation & Photometry Utilities
# ============================================================================

def flux_ujy_to_mag(flux_ujy):
    """
    Convert flux in μJy to AB magnitude; returns NaN for non-positive flux.
    
    Parameters
    ----------
    flux_ujy : float or np.ndarray
        Flux in microjansky
        
    Returns
    -------
    float or np.ndarray
        AB magnitude(s)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux_ujy / 3631e6)
    return mag


def flux_ujy_to_mag_err(flux_ujy, fluxerr_ujy):
    """
    Convert flux error to magnitude error using standard propagation.
    
    Parameters
    ----------
    flux_ujy : float or np.ndarray
        Flux in microjansky
    fluxerr_ujy : float or np.ndarray
        Flux error in microjansky
        
    Returns
    -------
    float or np.ndarray
        Magnitude error(s)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = (2.5 / np.log(10)) * (fluxerr_ujy / flux_ujy)
    return mag_err


def mag_to_flux_ujy(mag):
    """
    Convert AB magnitude to flux in μJy.
    
    Parameters
    ----------
    mag : float or np.ndarray
        AB magnitude(s)
        
    Returns
    -------
    float or np.ndarray
        Flux in microjansky
    """
    with np.errstate(over="ignore", invalid="ignore"):
        flux = 3631e6 * 10.0 ** (-mag / 2.5)
    return flux


def band_to_index(band_name, filter_short_list):
    """
    Resolve filter short name to filter index in list.
    
    Parameters
    ----------
    band_name : str or None
        Filter short name (e.g., 'VIS', 'NISP-H')
    filter_short_list : list
        List of filter short names
        
    Returns
    -------
    int
        index in filter_short_list or None if not found
    """
    if band_name is None:
        return None
    lookup = {name.lower(): i for i, name in enumerate(filter_short_list)}
    key = band_name.lower()
    if key not in lookup:
        raise ValueError(f"Band '{band_name}' not in filter list {filter_short_list}")
    return lookup[key]


# ============================================================================
# Filter & FITS Data Loading
# ============================================================================

def load_filter_metadata(filter_list_file, filt_dir="."):
    """
    Load filter metadata from filters_to_use.dat file.
    
    Expected format: 3-column file with filter_rel_path, short_name, col_stem
    (lines starting with # are comments)
    
    Parameters
    ----------
    filter_list_file : str
        Path to filters_to_use.dat file
    filt_dir : str
        Directory containing the filter files (default: current dir)
        
    Returns
    -------
    list of dict
        Each dict has keys: 'path', 'rel_path', 'short', 'col_stem'
    """
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


def load_photometry_from_fits(fits_path, filter_config, patch_id=98):
    """
    Load photometry and errors from COSMOS-Deep FITS catalog.
    
    Parameters
    ----------
    fits_path : str
        Path to COSMOS-Deep FITS catalog
    filter_config : list of tuple
        List of (short_name, col_stem) tuples for each filter
    patch_id : int
        Patch ID to select (default: 98)
        
    Returns
    -------
    tuple of (real_mag, real_sigma, real_det, filter_short)
        real_mag : (n_filt, n_gal) filter-major magnitude array
        real_sigma : (n_filt, n_gal) filter-major error array
        real_det : (n_filt, n_gal) boolean detection array
        filter_short : list of filter short names
    """
    from astropy.table import Table
    
    cat = Table.read(fits_path)
    
    # Select by patch ID
    patch_col = cat["patch_id_list"]
    try:
        patch_mask = np.array([int(v) == patch_id for v in patch_col])
    except (ValueError, TypeError):
        patch_mask = np.array([str(v).strip() == str(patch_id) for v in patch_col])
    
    cat = cat[patch_mask]
    
    # Extract magnitudes and errors for all filters
    n_filt = len(filter_config)
    n_gal = len(cat)
    
    real_mag = np.full((n_filt, n_gal), np.nan)
    real_sigma = np.full((n_filt, n_gal), np.nan)
    
    filter_short = []
    
    for fi, (fname, col_stem) in enumerate(filter_config):
        filter_short.append(fname)
        
        mag_col = col_stem
        err_col = col_stem + "_err"
        
        if mag_col in cat.colnames and err_col in cat.colnames:
            mag_vals = np.asarray(cat[mag_col], dtype=float)
            err_vals = np.asarray(cat[err_col], dtype=float)
            
            # Mark valid measurements
            valid = np.isfinite(mag_vals) & np.isfinite(err_vals) & (err_vals > 0)
            real_mag[fi, valid] = mag_vals[valid]
            real_sigma[fi, valid] = err_vals[valid]
    
    real_det = np.isfinite(real_mag)
    
    return real_mag, real_sigma, real_det, filter_short
