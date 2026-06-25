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
        
        zval = priors.sample_z_prior()
        massval = priors.sample_mass_prior()

        # Sample τ-delayed SFH parameters
        ti = np.random.uniform(0.0, cosmology.age(zval).value)[0] # Time when SF began, cosmic (Gyr)
        tau =  10**(np.random.uniform(np.log10(1e-2), np.log10(100)))  # Timescale of decrease (Gyr)

        # Generate SFH
        t = np.linspace(0, cosmology.age(zval).value, 1000)
        sfh, timeax = sfh_delayed_exponential(t, massval, tau, ti)
        sfh = sfh / 1e9  # Convert M☉/Gyr -> M☉/yr

        # Sample other parameters
        dust = priors.sample_Av_prior()
        met = priors.sample_Z_prior()
        
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
            sed = db.calc_fnu_sed_fast(spec_ujy, fcs[:, :, fc_index])

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
            filcurves, _, _ = db.make_filvalkit_simple(
                lam, zval, fkit_name=filter_list, filt_dir=filt_dir
            )
            sed = db.calc_fnu_sed_fast(spec_ujy, filcurves)
            return sed
    else:
        # Interpolate to given wavelength array
        from scipy.interpolate import interp1d
        interp_func = interp1d(lam, spec_ujy, bounds_error=False, fill_value=0)
        return interp_func(return_spec)