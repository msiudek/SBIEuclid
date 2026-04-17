"""
Galaxy simulation functions for SBIPIX training data generation.

This module uses FSPS (Conroy & Gunn 2010) and dense_basis (Iyer et al. 2019) for stellar population synthesis
and filter convolution. 
"""

import os
import numpy as np
import hickle
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
import dense_basis as db

from ..utils.sed_utils import sfh_delayed_exponential, convert_to_microjansky


def calc_fnu_sed_lam_weighted(fnuspec, filcurves, lam_z):
    """
    Filter integration with λ-weighting (photon-counting detector formula).

    Computes:  filvals[i] = Σ T(λ)·F_ν(λ)·λ  /  Σ T(λ)·λ

    This is the correct formula for CCD/NIR photon-counting detectors where the
    detector response is proportional to photon count (∝ λ) rather than energy.
    Compare to the unweighted formula used in dense_basis.calc_fnu_sed_fast:
        filvals[i] = Σ T(λ)·F_ν(λ)  /  Σ T(λ)   (energy-detector, uniform-grid)
    """
    filvals = np.zeros(filcurves.shape[1])
    for tindex in range(filcurves.shape[1]):
        nonzero = filcurves[:, tindex] > 0
        T = filcurves[nonzero, tindex]
        F = fnuspec[nonzero]
        L = lam_z[nonzero]
        denom = np.sum(T * L)
        if denom > 0:
            filvals[tindex] = np.sum(T * F * L) / denom
        else:
            filvals[tindex] = 0.0
    return filvals


def _to_scalar(value):
    """Convert scalar-like/array-like prior sample to a Python float."""
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Received empty prior sample")
    return float(arr.reshape(-1)[0])


def _ssfr_mass_slope_from_z(zval):
    """Piecewise mass-slope term B(z) for log-sSFR relation."""
    if zval < 1.0:
        return -0.08
    if zval < 2.0:
        return -0.15
    if zval < 3.0:
        return -0.22
    if zval < 4.0:
        return -0.28
    return -0.34


def _mean_log_ssfr(logmass, zval):
    """Mean log-sSFR model: μ(z) + B(z) * (logM - 10)."""
    mu_z = -10.0 + 0.8 * np.log10(1.0 + max(float(zval), 0.0))
    b_z = _ssfr_mass_slope_from_z(float(zval))
    return mu_z + b_z * (float(logmass) - 10.0)


def _sample_stellar_age_from_ssfr(target_log_ssfr, age_universe_gyr):
    """
    Sample stellar age (Gyr) anchored to inverse sSFR with scatter.

    The baseline uses t_age ~ 1/sSFR and applies log-normal scatter,
    clipped to a physically plausible fraction of the universe age.
    """
    ssfr_yr = 10 ** float(target_log_ssfr)
    if not np.isfinite(ssfr_yr) or ssfr_yr <= 0:
        return float(np.clip(0.3 * age_universe_gyr, 0.05, 0.9 * age_universe_gyr))

    age_from_ssfr_gyr = 1.0 / ssfr_yr / 1e9
    scatter_factor = 10 ** np.random.normal(0.0, 0.3)
    age_sample = age_from_ssfr_gyr * scatter_factor

    min_age = 0.05
    max_age = max(0.9 * float(age_universe_gyr), min_age)
    return float(np.clip(age_sample, min_age, max_age))


def _enforce_target_ssfr(sfh_gyr, timeax_gyr, logmass_target, target_log_ssfr, n_iter=3):
    """
    Iteratively enforce a target log-sSFR while keeping total formed mass fixed.

    Parameters
    ----------
    sfh_gyr : np.ndarray
        SFH in Msun/Gyr.
    timeax_gyr : np.ndarray
        Time axis in Gyr.
    logmass_target : float
        Target log10 formed stellar mass used to renormalize SFH integral.
    target_log_ssfr : float
        Target log10(sSFR/yr^-1).
    n_iter : int
        Number of correction iterations.
    """
    sfh = np.asarray(sfh_gyr, dtype=float).copy()
    timeax = np.asarray(timeax_gyr, dtype=float)
    if sfh.size == 0 or timeax.size == 0:
        return sfh

    recent_mask = timeax >= (np.max(timeax) - 0.1)  # last 100 Myr window
    if not np.any(recent_mask):
        recent_mask[-1] = True

    target_mass = 10 ** float(logmass_target)
    dt = np.gradient(timeax)

    for _ in range(max(int(n_iter), 1)):
        recent_sfr_yr = np.mean(sfh[recent_mask]) / 1e9
        if (not np.isfinite(recent_sfr_yr)) or (recent_sfr_yr <= 0):
            break

        current_log_ssfr = np.log10(recent_sfr_yr) - float(logmass_target)
        delta = float(target_log_ssfr) - current_log_ssfr

        scale_recent = np.clip(10 ** delta, 0.1, 10.0)
        sfh[recent_mask] *= scale_recent

        current_mass = np.sum(sfh * dt)
        if np.isfinite(current_mass) and current_mass > 0:
            sfh *= (target_mass / current_mass)

    return sfh


def generate_atlas_parametric(priors, N_pregrid=10, initial_seed=42, store=True, 
                             filter_list='filter_list.dat', filt_dir='filters/', 
                             norm_method='median', z_step=0.01, sp=None, 
                             cosmology=None, fname=None, path='pregrids/', 
                             lam_array_spline=[], rseed=None):
    """
    Generate a pregrid of galaxy properties and corresponding SEDs using parametric SFH.
    
    Uses τ-delayed star formation history: SFR(t) ∝ (t-t_i) * exp(-(t-t_i)/τ)
    Based on dense_basis framework (Iyer et al. 2019).
    
    Parameters
    ----------
    priors : dense_basis.Priors object
        Prior distributions for galaxy parameters
    N_pregrid : int, optional
        Number of SEDs in the pre-grid (default: 10)
    initial_seed : int, optional
        Initial seed for random number generation (default: 42)
    store : bool, optional
        Flag whether to store results or return as output (default: True)
    filter_list : str, optional
        File containing list of filter curves (default: 'filter_list.dat')
    filt_dir : str, optional
        Directory containing filter files (default: 'filters/')
    norm_method : str, optional
        Normalization for SEDs: 'none', 'max', 'median', 'area' (default: 'median')
    z_step : float, optional
        Step size in redshift for filter curve grid (default: 0.01)
    sp : fsps.StellarPopulation, optional
        FSPS stellar population object (default: None, will create one)
    cosmology : astropy.cosmology object, optional
        Cosmology object (default: None, will create FlatLambdaCDM)
    fname : str, optional
        Filename for saving (default: None, auto-generated)
    path : str, optional
        Directory for saving results (default: 'pregrids/')
    lam_array_spline : list, optional
        Wavelength array for spline interpolation (default: [])
    rseed : int, optional
        Random seed override (default: None)

    Returns
    -------
    dict or None
        If store=False, returns dictionary with simulated data.
        If store=True, saves to file and returns None.
        
    Notes
    -----
    The parametric SFH uses a τ-delayed model where star formation begins at
    cosmic time t_i and follows SFR(t) = (M/τ²)(t-t_i)exp(-(t-t_i)/τ).
    
    This function extends dense_basis.generate_atlas() to support parametric SFHs.
    
    Output dictionary contains:
    - 'zval': Redshift values
    - 'sfh_tuple': SFH parameters [M*, M*_formed, SFR, τ, t_i, Nparam]
    - 'mstar': Surviving stellar mass
    - 'sfr': Star formation rate
    - 'dust': Dust attenuation values
    - 'met': Metallicity values  
    - 'sed': Simulated SEDs
    """
    # Set up defaults
    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    
    if sp is None:
        import fsps
        sp = fsps.StellarPopulation(
            compute_vega_mags=False, zcontinuous=1, sfh=0, imf_type=1, 
            logzsol=0.0, dust_type=2, dust2=0.0, add_neb_emission=True
        )

    print('Generating atlas with:')
    print(f'N_pregrid: {N_pregrid}, Parametric SFH (delayed-tau model)')
    
    if rseed is not None:
        print(f'Setting random seed to: {rseed}')
        np.random.seed(rseed)

    # Initialize storage arrays
    zval_all = []
    sfh_tuple_all = []
    dust_all = []
    met_all = []
    sed_all = []
    mstar_all = []
    sfr_all = []

    Nparam = 2  # For parametric SFH

    for i in tqdm(range(int(N_pregrid)), desc="Generating parametric SEDs"):
        # Sample parameters from priors
        
        zval = _to_scalar(priors.sample_z_prior())
        massval = _to_scalar(priors.sample_mass_prior())
        age_gyr = float(cosmology.age(zval).value)

        target_log_ssfr = None
        if str(getattr(priors, 'sfr_prior_type', '')).lower() == 'ssfrlognormal':
            mean_log_ssfr = _mean_log_ssfr(massval, zval)
            target_log_ssfr = np.random.normal(mean_log_ssfr, 0.3)
            if hasattr(priors, 'ssfr_min') and hasattr(priors, 'ssfr_max'):
                target_log_ssfr = float(np.clip(target_log_ssfr, priors.ssfr_min, priors.ssfr_max))

        # Sample τ-delayed SFH parameters
        ti_max = max(age_gyr - 1e-3, 1e-3)
        if target_log_ssfr is None:
            ti = np.random.uniform(0.0, ti_max, size=1)[0]  # Time when SF began, cosmic (Gyr)
        else:
            stellar_age = _sample_stellar_age_from_ssfr(target_log_ssfr, age_gyr)
            ti = float(np.clip(age_gyr - stellar_age, 0.0, ti_max))
        tau =  10**(np.random.uniform(np.log10(1e-2), np.log10(100)))  # Timescale of decrease (Gyr)

        # Generate SFH
        t = np.linspace(0, cosmology.age(zval).value, 1000)
        sfh, timeax = sfh_delayed_exponential(t, massval, tau, ti)  # Msun/Gyr

        # For sSFRlognormal in parametric mode, enforce a physically motivated
        # mass- and redshift-dependent sSFR sequence while preserving total mass.
        if target_log_ssfr is not None:
            sfh = _enforce_target_ssfr(sfh, timeax, massval, target_log_ssfr, n_iter=3)

        sfh = sfh / 1e9  # Convert M☉/Gyr -> M☉/yr for FSPS tabular SFH

        # Sample other parameters
        dust = _to_scalar(priors.sample_Av_prior())
        met = _to_scalar(priors.sample_Z_prior())
        
        # Ensure SFH is valid
        sfh = np.where(np.isnan(sfh) | (sfh < 1e-33), 1.1e-33, sfh)

        # Generate spectrum
        specdetails = [sfh, timeax, dust, met, zval]

        if len(lam_array_spline) > 0:
            sed = makespec_parametric(
                specdetails, priors, sp, cosmology, filter_list, 
                filt_dir, return_spec=lam_array_spline, peraa=True
            )
        else:
            # Generate full spectrum first time to set up filter grid
            lam, spec_ujy = makespec_parametric(
                specdetails, priors, sp, cosmology, filter_list, 
                filt_dir, return_spec=True
            )

            if i == 0:
                # Create filter transmission curve grid for faster computation
                fc_zgrid = np.arange(
                    priors.z_min - z_step, 
                    priors.z_max + z_step, 
                    z_step
                )
                
                temp_fc, temp_lz, temp_lz_lores = db.make_filvalkit_simple(
                    lam, priors.z_min, fkit_name=filter_list, filt_dir=filt_dir
                )

                fcs = np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
                lzs = np.zeros((temp_lz.shape[0], len(fc_zgrid)))
                lzs_lores = np.zeros((temp_lz_lores.shape[0], len(fc_zgrid)))

                for j in range(len(fc_zgrid)):
                    fcs[:, :, j], lzs[:, j], lzs_lores[:, j] = db.make_filvalkit_simple(
                        lam, fc_zgrid[j], fkit_name=filter_list, filt_dir=filt_dir
                    )

            # Use pre-computed filter grid
            fc_index = np.argmin(np.abs(zval - fc_zgrid))
            sed = calc_fnu_sed_lam_weighted(spec_ujy, fcs[:, :, fc_index], lzs[:, fc_index])

        # Normalization
        norm_fac = 1.0
        sed = sed / norm_fac
        mstar = np.log10(sp.stellar_mass / norm_fac)
        mformed = np.log10(sp.formed_mass / norm_fac)
        sfr = np.log10(np.mean(sfh[-100:]))  # Averaged over last 100 Myr

        # Store SFH parameters
        sfh_tuple = np.array([mstar, mformed, sfr, tau, ti, Nparam])

        # Append to lists
        zval_all.append(zval)
        sfh_tuple_all.append(sfh_tuple)
        dust_all.append(dust)
        met_all.append(met)
        sed_all.append(sed)
        mstar_all.append(mstar)
        sfr_all.append(sfr)

    # Create output dictionary
    pregrid_dict = {
        'zval': np.array(zval_all),
        'sfh_tuple': np.array(sfh_tuple_all),
        'mstar': np.array(mstar_all), 
        'sfr': np.array(sfr_all),
        'dust': np.array(dust_all), 
        'met': np.array(met_all),
        'sed': np.array(sed_all)
    }

    if store:
        if fname is None:
            fname = 'sfh_pregrid_size'
            
        if os.path.exists(path):
            print(f'Path exists. Saved atlas at: {path}{fname}_{N_pregrid}_Nparam_{Nparam}.dbatlas')
        else:
            os.mkdir(path)
            print(f'Created directory and saved atlas at: {path}{fname}_{N_pregrid}_Nparam_{Nparam}.dbatlas')
        
        try:
            hickle.dump(
                pregrid_dict,
                f'{path}{fname}_{N_pregrid}_Nparam_{Nparam}.dbatlas',
                compression='gzip', 
                compression_opts=9
            )
        except:
            print('Storing without compression')
            hickle.dump(
                pregrid_dict,
                f'{path}{fname}_{N_pregrid}_Nparam_{Nparam}.dbatlas'
            )
        
        return None
    else:
        return pregrid_dict


def makespec_parametric(specdetails, priors, sp, cosmo, filter_list=[], 
                       filt_dir=[], return_spec=False, peraa=False, input_sfh=False):
    """
    Generate spectrum or SED from physical parameters using parametric SFH.
    
    Uses dense_basis framework for filter convolution and spectral processing.

    Parameters
    ----------
    specdetails : list
        If input_sfh=False: [sfh_tuple, dust, met, zval]
        If input_sfh=True: [sfh, timeax, dust, met, zval]
    priors : dense_basis.Priors object
        Prior distributions object
    sp : fsps.StellarPopulation
        FSPS stellar population object
    cosmo : astropy.cosmology object
        Cosmology object
    filter_list : list, optional
        List of filter files (default: [])
    filt_dir : list, optional
        Filter directory (default: [])
    return_spec : bool or np.ndarray, optional
        If True: return full spectrum
        If False: return photometric SED
        If array: return spectrum interpolated to given wavelengths (default: False)
    peraa : bool, optional
        Return spectrum per Angstrom (default: False)
    input_sfh : bool, optional
        Whether SFH is provided directly (default: False)

    Returns
    -------
    Various
        Depending on return_spec:
        - If True: (wavelength, spectrum) tuple
        - If False: photometric SED array
        - If array: interpolated spectrum
        
    Notes
    -----
    This function uses dense_basis for filter convolution
    """
    # Configure FSPS parameters
    sp.params['sfh'] = 3  # Tabular SFH
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1  # Chabrier

    # Extract parameters
    [sfh, tax, dust, met, zval] = specdetails
    sp.params['dust2'] = dust
    sp.params['logzsol'] = met
    sp.params['gas_logz'] = met  # Match stellar to gas-phase metallicity
    sp.params['zred'] = zval
    
    # Ensure SFH is valid
    sfh = np.where(np.isnan(sfh) | (sfh < 1e-33), 1e-33, sfh)
    sp.set_tabular_sfh(tax, sfh)

    # Generate spectrum
    # Add small time offset to get latest SSPs
    lam, spec = sp.get_spectrum(tage=cosmo.age(zval).value + 1e-4, peraa=peraa)
    spec_ujy = convert_to_microjansky(spec, zval, cosmo)

    # Return based on return_spec parameter
    if isinstance(return_spec, bool):
        if return_spec:
            return lam, spec_ujy
        else:
            # Generate photometric SED using dense_basis
            filcurves, lam_z_filt, _ = db.make_filvalkit_simple(
                lam, zval, fkit_name=filter_list, filt_dir=filt_dir
            )
            sed = calc_fnu_sed_lam_weighted(spec_ujy, filcurves, lam_z_filt)
            return sed
    else:
        # Interpolate to given wavelength array
        from scipy.interpolate import interp1d
        interp_func = interp1d(lam, spec_ujy, bounds_error=False, fill_value=0)
        return interp_func(return_spec)