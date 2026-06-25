"""
SED processing utilities for SBIPIX.
"""

import numpy as np


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
    A = 10**logmassval / integral_result[0]
    
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
    
    # Convert to flux at Earth: F = L / (4π d_L²)
    flux_erg = spec_erg / (4 * np.pi * d_L**2)
    
    # Convert to Jansky (1 Jy = 1e-23 erg/s/cm²/Hz)
    flux_jy = flux_erg / 1e-23
    
    # Convert to microjansky
    flux_ujy = flux_jy * 1e6
    
    return flux_ujy