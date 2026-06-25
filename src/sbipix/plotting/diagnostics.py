"""
Diagnostic plotting functions for SBIPIX.

This module contains plotting functions that work with SBIPIX class instances
to create visualizations for model diagnostics and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
import fsps
import dense_basis as db
from dense_basis import make_filvalkit_simple, calc_fnu_sed_fast
from sbipix.utils.sed_utils import mag_conversion
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from tqdm import tqdm, trange
import os
import h5py
import glob
from matplotlib.lines import Line2D

# Set matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['font.size'] = 10
mpl.rcParams["figure.figsize"] = (6, 5)
mpl.rcParams["mathtext.fontset"] = 'dejavuserif'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

def plot_filters(sbipix_model):
    """
    Plot the filter transmission curves.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with filter configuration
    """
    db.plot_filterset(
        filter_list=sbipix_model.filter_list, 
        filt_dir=sbipix_model.filter_path
    )


def plot_theta(sbipix_model, bins=50, limit_sfr=False, range_sfr=(-10, 2), 
               figsize=(12, 8), save=False, filename=None):
    """
    Plot histograms of the physical parameters.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with simulation data
    bins : int, optional
        Number of histogram bins (default: 50)
    limit_sfr : bool, optional
        Whether to limit SFR histogram range (default: False)
    range_sfr : tuple, optional
        Range for SFR histogram if limit_sfr=True (default: (-10, 2))
    figsize : tuple, optional
        Figure size (default: (12, 8))
    save : bool, optional
        Whether to save plots (default: False)
    filename : str, optional
        Base filename for saving (default: None)
    """
    if sbipix_model.theta is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    n_params = len(sbipix_model.theta[0, :])
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i in range(n_params):
        ax = axes[i]
        ix = 2 if sbipix_model.both_masses else 1
        
        if i == ix and limit_sfr:
            ax.hist(sbipix_model.theta[:, i], bins=bins, range=range_sfr, 
                   alpha=0.7, edgecolor='black')
        else:
            ax.hist(sbipix_model.theta[:, i], bins=bins, 
                   alpha=0.7, edgecolor='black')
        
        ax.set_xlabel(sbipix_model.labels[i], fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(sbipix_model.theta[:, i])
        std_val = np.std(sbipix_model.theta[:, i])
        ax.text(0.05, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        save_name = filename or 'parameter_histograms.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()


def plot_fluxes(sbipix_model, n, figsize=(10, 6), save=False, filename=None,
                show_individual=True, show_median=True):
    """
    Plot the SEDs for the first n simulated galaxies.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with simulation data
    n : int
        Number of galaxies to plot
    figsize : tuple, optional
        Figure size (default: (10, 6))
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    show_individual : bool, optional
        Whether to show individual SEDs (default: True)
    show_median : bool, optional
        Whether to show median SED (default: True)
    """
    if sbipix_model.obs is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    fluxes = mag_conversion(sbipix_model.obs, convert_to='flux')
    lam_eff = np.load('./observational_properties/lam_eff.npy')[:]
    
    plt.figure(figsize=figsize)
    
    # Plot individual SEDs
    if show_individual:
        for i in range(min(n, len(fluxes))):
            plt.plot(lam_eff, fluxes[i], '.', alpha=0.6, markersize=4,
                    label=f'Galaxy {i+1}' if n <= 5 else None)
    
    # Plot median SED
    if show_median and n > 1:
        median_flux = np.median(fluxes[:n], axis=0)
        plt.plot(lam_eff, median_flux, 'ro-', linewidth=2, markersize=6,
                label=f'Median (n={n})')
    
    plt.yscale('log')
    plt.ylabel('Flux density [$\\mu$Jy]', fontsize=14)
    plt.xlabel('Wavelength [$\\AA$]', fontsize=14)
    plt.title(f'Simulated SEDs (showing {min(n, len(fluxes))} galaxies)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    if (show_individual and n <= 5) or show_median:
        plt.legend()
    
    if save:
        save_name = filename or f'seds_n{n}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()


def plot_test_performance(sbipix_model, n_test=1000, n_theta=None, save=False, 
                         name=None, figsize=(8, 8)):
    """
    Plot test performance comparing true vs predicted values.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with test results
    n_test : int, optional
        Number of test samples to plot (default: 1000)
    n_theta : int, optional
        Number of parameters to plot (default: auto-detect)
    save : bool, optional
        Whether to save plots (default: False)
    name : str, optional
        Custom filename prefix for saving (default: None)
    figsize : tuple, optional
        Figure size for each plot (default: (8, 8))
    """
    if sbipix_model.means_test is None or sbipix_model.theta is None:
        raise ValueError("No test results found. Run test_performance() first.")
    
    if n_theta is None:
        n_theta = 8 if sbipix_model.infer_z else 7

    for i in range(n_theta):
        p_true = sbipix_model.theta[:n_test, i]
        p_pred = sbipix_model.means_test[:, i]
        
        g = sns.jointplot(x=p_true, y=p_pred, height=figsize[0], 
                         label='Test data', legend=False)
        
        # Add perfect prediction line
        x_range = np.arange(np.min(p_true), np.max(p_true), 0.01)
        g.plot_joint(sns.kdeplot, color="k", zorder=1, levels=6, label='KDE')
        g.ax_joint.plot(x_range, x_range, '-r', label='y=x', linewidth=2)
        
        # Set labels
        g.set_axis_labels(
            sbipix_model.labels[i] + ' (true)', 
            sbipix_model.labels[i] + ' (predicted)', 
            fontsize=16
        )
        
        # Add R² score
        r2_score = sm.r2_score(p_true, p_pred)
        g.ax_joint.text(
            0.05, 0.95, f"R² = {r2_score:.3f}", 
            transform=g.ax_joint.transAxes,
            fontsize=14, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add RMSE
        rmse = np.sqrt(np.mean((p_true - p_pred)**2))
        g.ax_joint.text(
            0.05, 0.88, f"RMSE = {rmse:.3f}", 
            transform=g.ax_joint.transAxes,
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        if save:
            if name is None:
                suffix = 'tau' if sbipix_model.parametric else 'dirichlet'
                filename = f'./plots/test_performance_{suffix}_{i}.pdf'
            else:
                filename = f'./plots/{name}{i}.pdf'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved plot to {filename}")
        
        plt.show()


def plot_performance_obs_from_catalog(sbipix_model, parameter='z', save=False, 
                                     filename=None, figsize=(8, 6)):
    """
    Plot performance on observational catalog.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance with observational results
    parameter : str, optional
        Which parameter to plot (default: 'z' for redshift)
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Custom filename (default: None)
    figsize : tuple, optional
        Figure size (default: (8, 6))
    """
    print(f"Plotting performance for parameter: {parameter}")
    
    plt.figure(figsize=figsize)
    
    # This would be implemented based on available observational data
    # Placeholder for now
    plt.text(0.5, 0.5, f'Observational performance plot for {parameter}\n'
                       'Implementation depends on available catalog data',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.title(f'Observational Performance: {parameter}', fontsize=16)
    
    if save:
        save_name = filename or f'obs_performance_{parameter}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {save_name}")
    
    plt.show()


def plot_corner(sbipix_model, n_samples=1000, parameters=None, save=False, 
                filename=None):
    """
    Create corner plot of posterior samples.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance
    n_samples : int, optional
        Number of samples to plot (default: 1000)
    parameters : list, optional
        Which parameters to include (default: None, uses all)
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    """
    try:
        import corner
    except ImportError:
        print("corner package not available. Install with: pip install corner")
        return
    
    if sbipix_model.theta is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    # Select parameters to plot
    if parameters is None:
        data = sbipix_model.theta[:n_samples]
        labels = sbipix_model.labels
    else:
        param_indices = [sbipix_model.labels.index(p) for p in parameters]
        data = sbipix_model.theta[:n_samples, param_indices]
        labels = parameters
    
    # Create corner plot
    fig = corner.corner(data, labels=labels, show_titles=True, 
                       title_kwargs={"fontsize": 12})
    
    if save:
        save_name = filename or 'corner_plot.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved corner plot to {save_name}")
    
    plt.show()


def plot_sed_comparison(sbipix_model, galaxy_idx, model_sed=None, 
                       figsize=(10, 6), save=False, filename=None):
    """
    Compare observed and model SEDs for a specific galaxy.
    
    Parameters
    ----------
    sbipix_model : SBIPIX
        SBIPIX model instance
    galaxy_idx : int
        Index of galaxy to plot
    model_sed : np.ndarray, optional
        Model SED to compare (default: None)
    figsize : tuple, optional
        Figure size (default: (10, 6))
    save : bool, optional
        Whether to save plot (default: False)
    filename : str, optional
        Filename for saving (default: None)
    """
    if sbipix_model.obs is None:
        raise ValueError("No simulation data found. Run load_simulation() first.")
    
    lam_eff = np.load('obs/obs_properties/lam_eff.npy')[:]
    obs_flux = mag_conversion(sbipix_model.obs[galaxy_idx], convert_to='flux')
    
    plt.figure(figsize=figsize)
    
    # Plot observed SED
    plt.errorbar(lam_eff, obs_flux, fmt='o', markersize=6, 
                label='Simulated Data', capsize=3)
    
    # Plot model SED if provided
    if model_sed is not None:
        plt.plot(lam_eff, model_sed, 's-', linewidth=2, markersize=4,
                label='Best-fit Model')
    
    plt.yscale('log')
    plt.xlabel('Wavelength [Å]', fontsize=14)
    plt.ylabel('Flux density [μJy]', fontsize=14)
    plt.title(f'SED Comparison - Galaxy {galaxy_idx}', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save:
        save_name = filename or f'sed_comparison_gal{galaxy_idx}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save_name}")
    
    plt.show()

"""
Posterior Predictive Check for SBIPIX
"""

def flux2mag(x):
    """Convert flux in microjansky to AB magnitude"""
    return -2.5 * np.log10(x * 1e-6 / 3631)

def mag2flux(x):
    """Convert AB magnitude to flux in microjansky"""
    return (3631 / 1e-6) * 10 ** (x / (-2.5))

def make_csp_spectrum_sbipix(theta, cosmo, sp):
    """
    Generate CSP spectrum for SBIPIX parameters
    
    Parameters:
    -----------
    theta : array
        Parameter array [log_total_mass, log_sfr_inst, tau, ti, met, dust]
    cosmo : astropy.cosmology
        Cosmology object
    sp : fsps.StellarPopulation
        FSPS stellar population object
        
    Returns:
    --------
    lam : array
        Wavelength array
    spec_csp : array
        Spectrum in microjansky
    """
    
    log_total_mass, log_sfr_inst, tau, ti, zval, met, dust = theta
    
    sp.params['sfh'] = 4  # tau-delayed SFH
    sp.params['tau'] = tau
    sp.params['sf_start'] = ti
    sp.params['dust2'] = dust
    sp.params['logzsol'] = met
    
    lam, spec_csp = sp.get_spectrum(tage=cosmo.age(zval).value)
    
    # Convert to microjansky
    from dense_basis import convert_to_microjansky, calc_fnu_sed_fast
    spec_csp_ujy = convert_to_microjansky(spec_csp, zval, cosmo)
    
    return lam, spec_csp_ujy * 10**log_total_mass

def make_spec_plot_sbipix(lam, spec, lam_min=4000, lam_max=55000, 
                         theta=None, alpha=1, c='tab:blue', 
                         label='SED template', add_mags=False, ax=None):
    """
    Spectrum plotting utility for SBIPIX
    
    Parameters:
    -----------
    lam : array
        Wavelength array
    spec : array
        Spectrum array
    lam_min, lam_max : float
        Wavelength limits in Angstroms
    theta : array
        Parameter array (optional)
    alpha : float
        Transparency
    c : str
        Color
    label : str
        Plot label
    add_mags : bool
        Whether to add magnitude axis
    ax : matplotlib.axes
        Axes object
        
    Returns:
    --------
    ax : matplotlib.axes
        Axes object
    ax2 : matplotlib.axes (optional)
        Secondary axis if add_mags=True
    """
    
    if ax is None:
        ax = plt.gca()
        
    lam_mask = (lam > lam_min) & (lam < lam_max)
    ax.plot(lam[lam_mask]/1e4, spec[lam_mask], label=label, alpha=alpha, c=c)
    
    ax.set_xticks(ticks=[0.5, 1, 2, 5], labels=[0.5, 1, 2, 5])
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if add_mags:
        ax2 = ax.secondary_yaxis('right', functions=(flux2mag, mag2flux))
        ax2.set_ylabel('Mag')
        ax2.set_yticks(ticks=[27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                      labels=[27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
        ax2.invert_yaxis()
        return ax, ax2
    else:
        return ax

def plot_posterior_templates_sbipix(sx, galaxy_id, phot, err, z, posteriors, sp=None,
                                   n_samples=500, pixel_type='high_sn',
                                   ax=None, add_mags=False, show_background=True,
                                   limits_file='obs/obs_properties/background_noise_hainline.npy',
                                   path_obs_properties='obs/obs_properties/'):
    """
    Plot posterior predictive templates for SBIPIX
    
    Parameters:
    -----------
    sx : sbipix
        SBIPIX instance
    galaxy_id : int
        Galaxy ID
    phot : array
        Photometry array
    err : array
        Error array
    z : float
        Redshift
    posteriors : array
        Posterior samples array
    sp : fsps.StellarPopulation, optional
        FSPS stellar population instance (default: None, will initialize if None)
    n_samples : int
        Number of samples to plot
    pixel_type : str
        Type of pixel ('high_sn', 'random', etc.)
    ax : matplotlib.axes
        Axes object
    add_mags : bool
        Whether to add magnitude axis
    show_background : bool
        Whether to show background limits
    limits_file : str
        Path to background limits file
    path_obs_properties : str
        Path to observational properties directory
        
    Returns:
    --------
    ax : matplotlib.axes
        Axes object
    best_theta : array
        Best-fit parameters
    """

    # If not initialized, create a new StellarPopulation instance
    if sp is None:
        import fsps
        sp = fsps.StellarPopulation(
            compute_vega_mags=False,
            zcontinuous=1,
            sfh=0,  # Simple stellar populations
            dust_type=2,  # Calzetti dust attenuation
            imf_type=1,  # Chabrier IMF
            add_neb_continuum=True,
            add_neb_emission=True
        )
        
    # Load cosmology
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Load effective wavelengths
    lam_eff_file = path_obs_properties + 'lam_eff.npy'
    if os.path.exists(lam_eff_file):
        lam_eff = np.load(lam_eff_file)[:19] / 1e4
    else:
        # Default JADES effective wavelengths (in microns)
        lam_eff = np.array([0.90, 1.15, 1.50, 1.82, 2.00, 2.10, 2.77, 3.35, 3.56, 
                           4.10, 4.30, 4.44, 4.60, 4.80, 0.435, 0.606, 0.775, 0.814, 0.850])
    
    # Load background limits if available
    if hasattr(sx, 'limits') or show_background:
        try:
            limits = np.load(limits_file)
        except:
            limits = None
    else:
        limits = None
    
    # Apply limits to observations
    obs_sed = np.copy(phot)
    obs_err = np.copy(err)
    
    if limits is not None:
        for i in range(len(limits)):
            if obs_sed[i] < limits[i]:
                obs_sed[i] = limits[i]
                obs_err[i] = np.nan
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot posterior samples
    losses = []
    specs = []
    
    n_plot = min(n_samples, len(posteriors))
    
    for k in trange(n_plot, desc="Plotting posterior samples"):
        
        # Extract parameters (assuming SBIPIX tau model)
        if posteriors.shape[-1] == 7:  # [log_M*, log_M*_formed, log_SFR, tau, ti, [M/H], Av]
            log_total_mass = posteriors[k, 0]
            log_sfr_inst = posteriors[k, 2] 
            tau = posteriors[k, 3]
            ti = posteriors[k, 4]
            met = posteriors[k, 5]
            dust = posteriors[k, 6]
        else:
            print(f"Warning: Unexpected posterior shape {posteriors.shape}")
            continue
            
        theta = (log_total_mass, log_sfr_inst, tau, ti, z, met, dust)
        
        try:
            # Generate spectrum
            lam, spec_csp_ujy = make_csp_spectrum_sbipix(theta, cosmo, sp)
            
            # Calculate photometry
            filcurves, _, _ = make_filvalkit_simple(lam, z, 
                                                  fkit_name='filters_jades_no_wfc.dat',
                                                  filt_dir=path_obs_properties)
            sed_csp_ujy = calc_fnu_sed_fast(spec_csp_ujy, filcurves)
            
            # Plot spectrum
            if k == 0:
                if add_mags:
                    ax, ax2 = make_spec_plot_sbipix(lam*(1+z), spec_csp_ujy, 
                                                   theta=theta, alpha=0.05, 
                                                   add_mags=add_mags, ax=ax)
                else:
                    ax = make_spec_plot_sbipix(lam*(1+z), spec_csp_ujy, 
                                              theta=theta, alpha=0.05, ax=ax)
            else:
                ax = make_spec_plot_sbipix(lam*(1+z), spec_csp_ujy, 
                                          theta=theta, alpha=0.05, ax=ax)
            
            # Plot photometry
            ax.plot(lam_eff, sed_csp_ujy, 'o', markersize=8, alpha=0.05,
                   markerfacecolor="None", markeredgecolor='tab:blue')
            
            # Calculate loss
            loss = []
            for i in range(len(obs_sed)):
                if np.isfinite(obs_err[i]):
                    loss.append(np.abs(sed_csp_ujy[i] - obs_sed[i]) / obs_err[i])
                else:
                    loss.append(0)
            
            losses.append(np.sum(loss))
            specs.append(spec_csp_ujy)
            
        except Exception as e:
            print(f"Error processing sample {k}: {e}")
            continue
    
    # Plot observations
    for i in range(len(obs_sed)):
        if np.isfinite(obs_err[i]):
            ax.errorbar(lam_eff[i], obs_sed[i], yerr=obs_err[i], 
                       fmt='ko', ecolor='k', markersize=7, alpha=1)
    
    # Plot background limits
    if show_background and limits is not None:
        ax.plot(lam_eff, limits, 'v', markerfacecolor="None", 
               markeredgecolor='k', markersize=10, 
               label='σ_background', alpha=1.0)
    else:
        if limits is not None:
            for i in range(len(obs_sed)):
                if not np.isfinite(obs_err[i]):
                    ax.plot(lam_eff[i], limits[i], 'v', markerfacecolor="k",
                           markeredgecolor='k', markersize=12, alpha=1)
    
    # Find and plot best fit
    if losses:
        best_idx = np.argmin(losses)
        best_theta = posteriors[best_idx]
        
        # Plot best fit
        log_total_mass = best_theta[0]
        log_sfr_inst = best_theta[2] 
        tau = best_theta[3]
        ti = best_theta[4]
        met = best_theta[5]
        dust = best_theta[6]
        
        theta_best = (log_total_mass, log_sfr_inst, tau, ti, z, met, dust)
        lam, spec_best = make_csp_spectrum_sbipix(theta_best, cosmo, sp)
        
        ax = make_spec_plot_sbipix(lam*(1+z), spec_best, theta=theta_best,
                                  alpha=1, c='tab:orange', 
                                  label='Best fit', ax=ax)
        
        # Calculate and plot best photometry
        filcurves, _, _ = make_filvalkit_simple(lam, z, 
                                              fkit_name='filters_jades_no_wfc.dat',
                                              filt_dir=path_obs_properties)
        sed_best = calc_fnu_sed_fast(spec_best, filcurves)
        ax.plot(lam_eff, sed_best, 'o', markersize=10, alpha=0.8,
               markerfacecolor="None", markeredgecolor='tab:orange')
        
        # Plot median spectrum
        if specs:
            specs = np.array(specs)
            mean_spec = np.mean(specs, axis=0)
            ax = make_spec_plot_sbipix(lam*(1+z), mean_spec, add_mags=False,
                                      c='m', alpha=1, label='Median', ax=ax)
            
            sed_median = calc_fnu_sed_fast(mean_spec, filcurves)
            ax.plot(lam_eff, sed_median, 'o', markersize=10, alpha=1,
                   markerfacecolor="None", markeredgecolor='m')
    else:
        best_theta = None
    
    # Set axis properties
    ax.set_xticks(ticks=[0.5, 1, 2, 5], labels=[0.5, 1, 2, 5])
    ax.set_xlabel("Observed wavelength [μm]")
    ax.set_ylabel("Flux [μJy]")
    
    # Add rest-frame wavelength axis
    original_ticks = [0.5, 1, 2, 5]
    restframe_ticks = [tick / (1 + z) for tick in original_ticks]
    
    axis_rf = ax.twiny()
    axis_rf.set_xscale("log")
    axis_rf.set_xlim(ax.get_xlim())
    axis_rf.set_xticks(original_ticks)
    axis_rf.set_xticklabels([f"{tick:.1f}" for tick in restframe_ticks])
    axis_rf.set_xlabel(r"Rest-frame wavelength [μm]")
    
    if add_mags:
        return ax, ax2, best_theta
    else:
        return ax, best_theta


def posterior_predictive_check_sbipix(sx, data_file, galaxy_ids=None, 
                                     pixel_selections=['max_snr', 'random'],
                                     sp=None,
                                     n_samples=500, save_fig=True, 
                                     output_dir='./posterior_checks/',
                                     filters_sn=[6, 8, 11],
                                     posterior_group='posterior_tau',
                                     limits_file='obs/obs_properties/background_noise_hainline.npy',
                                     path_obs_properties='obs/obs_properties/'):
    """
    Run posterior predictive checks for SBIPIX galaxies with flexible pixel selection
    
    Parameters:
    -----------
    sx : sbipix
        SBIPIX instance 
    data_file : str
        Path to HDF5 file with posteriors
    galaxy_ids : list or None
        List of galaxy IDs to process
    pixel_selections : list
        List of selection methods: 'max_snr', 'random', or integer pixel indices
    sp : fsps.StellarPopulation, optional
        FSPS stellar population instance (default: None, will initialize if None)
    n_samples : int
        Number of posterior samples to use
    save_fig : bool
        Whether to save figures
    output_dir : str
        Output directory for figures
    filters_sn : list
        Filter indices for S/N calculation [F277W, F356W, F444W by default]
    posterior_group : str
        Name of posterior group to use
    limits_file : str
        Path to background limits file
    path_obs_properties : str
        Path to observational properties directory
    """
    
    # Initialize StellarPopulation if not provided
    if sp is None:
        import fsps
        sp = fsps.StellarPopulation(
            compute_vega_mags=False,
            zcontinuous=1,
            sfh=0,  # Simple stellar populations
            dust_type=2,  # Calzetti dust attenuation
            imf_type=1,  # Chabrier IMF
            add_neb_continuum=True,
            add_neb_emission=True
        )
    
    # Create output directory
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(data_file, "r") as hdf:
        
        if galaxy_ids is None:
            galaxy_ids = [int(gid) for gid in hdf["galaxies"].keys()]
        
        print(f"Processing {len(galaxy_ids)} galaxies: {galaxy_ids}")
        print(f"Pixel selections: {pixel_selections}")
        
        # Set up figure
        n_cols = len(pixel_selections)
        n_rows = len(galaxy_ids)
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(8*n_cols, 6*n_rows), 
                                sharex=True, sharey=False)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        
        for i, galaxy_id in enumerate(galaxy_ids):
            z = hdf[f"galaxies/{galaxy_id}"].attrs["redshift"]
            
            for j, pixel_selection in enumerate(pixel_selections):
                
                print(f"\nProcessing galaxy {galaxy_id}, selection: {pixel_selection}")
                
                ax = axes[i, j]
                
                try:
                    # Get galaxy data
                    galaxy = hdf[f"galaxies/{galaxy_id}"]
                    coordinates = galaxy["coordinates"][:]
                    all_photometry = galaxy["photometry"][:]
                    all_errors = galaxy["error"][:]
                    
                    # Find posterior group
                    available_groups = [g for g in galaxy.keys() if 'posterior' in g]
                    if posterior_group not in galaxy and available_groups:
                        posterior_group = available_groups[0]
                        print(f"Using posterior group: {posterior_group}")
                    
                    posteriors_data = galaxy[posterior_group]
                    fitted_pixels = posteriors_data['pixels_fitted'][:]
                    posteriors = posteriors_data['posteriors'][:]
                    
                    # Select pixel based on method - ONLY from fitted pixels
                    if pixel_selection == 'max_snr':
                        # Calculate S/N only for fitted pixels
                        snr_values = []
                        
                        for k in range(len(fitted_pixels)):
                            fitted_pixel_idx = fitted_pixels[k]
                            pixel_snr = []
                            for filt_idx in filters_sn:
                                if filt_idx < all_photometry.shape[1]:
                                    flux = all_photometry[fitted_pixel_idx, filt_idx]
                                    error = all_errors[fitted_pixel_idx, filt_idx]
                                    if np.isfinite(flux) and np.isfinite(error) and error > 0:
                                        pixel_snr.append(flux / error)
                            
                            if len(pixel_snr) > 0:
                                snr_values.append(np.mean(pixel_snr))
                            else:
                                snr_values.append(np.nan)
                        
                        if len(snr_values) > 0:
                            snr_values = np.array(snr_values)
                            max_snr_idx = np.nanargmax(snr_values)  # Index in fitted_pixels array
                            selected_pixel_idx = fitted_pixels[max_snr_idx]  # Actual pixel index
                            snr_val = snr_values[max_snr_idx]
                            selection_label = "Max S/N"
                            
                            # Get photometry and error for the selected fitted pixel
                            phot = all_photometry[selected_pixel_idx]
                            err = all_errors[selected_pixel_idx]
                            
                            # Get corresponding posteriors
                            pixel_posteriors = posteriors[max_snr_idx]
                            fitted_idx = max_snr_idx
                            
                        else:
                            raise ValueError("No valid S/N values found among fitted pixels")
                        
                    elif pixel_selection == 'random':
                        # Select random index from fitted pixels
                        if len(fitted_pixels) > 0:
                            random_fitted_idx = np.random.randint(0, len(fitted_pixels))
                            selected_pixel_idx = fitted_pixels[random_fitted_idx]
                            
                            # Get photometry and error for the selected fitted pixel
                            phot = all_photometry[selected_pixel_idx]
                            err = all_errors[selected_pixel_idx]
                            
                            # Get corresponding posteriors
                            pixel_posteriors = posteriors[random_fitted_idx]
                            fitted_idx = random_fitted_idx
                            
                            # Calculate S/N for selected pixel
                            pixel_snr = []
                            for filt_idx in filters_sn:
                                if filt_idx < all_photometry.shape[1]:
                                    flux = phot[filt_idx]
                                    error = err[filt_idx]
                                    if np.isfinite(flux) and np.isfinite(error) and error > 0:
                                        pixel_snr.append(flux / error)
                            
                            snr_val = np.mean(pixel_snr) if len(pixel_snr) > 0 else np.nan
                            selection_label = "Random"
                        else:
                            raise ValueError("No fitted pixels available")
                            
                    elif isinstance(pixel_selection, (int, np.integer)):
                        # Direct index into fitted pixels array
                        if pixel_selection >= len(fitted_pixels):
                            raise ValueError(f"Fitted pixel index {pixel_selection} out of range (max: {len(fitted_pixels)-1})")
                        
                        selected_pixel_idx = fitted_pixels[pixel_selection]
                        
                        # Get photometry and error for the selected fitted pixel
                        phot = all_photometry[selected_pixel_idx]
                        err = all_errors[selected_pixel_idx]
                        
                        # Get corresponding posteriors
                        pixel_posteriors = posteriors[pixel_selection]
                        fitted_idx = pixel_selection
                        
                        # Calculate S/N for selected pixel
                        pixel_snr = []
                        for filt_idx in filters_sn:
                            if filt_idx < all_photometry.shape[1]:
                                flux = phot[filt_idx]
                                error = err[filt_idx]
                                if np.isfinite(flux) and np.isfinite(error) and error > 0:
                                    pixel_snr.append(flux / error)
                        
                        snr_val = np.mean(pixel_snr) if len(pixel_snr) > 0 else np.nan
                        selection_label = f"Fitted Pixel {pixel_selection}"
                    
                    else:
                        raise ValueError(f"Unknown pixel_selection: {pixel_selection}. Use 'max_snr', 'random', or integer pixel index.")
                    
                    print(f"Selected fitted pixel {selected_pixel_idx} (fitted index {fitted_idx}) with S/N = {snr_val:.2f}")
                    
                except Exception as e:
                    print(f"Error getting pixel data for {galaxy_id}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    continue
                
                # Plot posterior predictive check
                try:
                    ax, ax2, best_theta = plot_posterior_templates_sbipix(
                        sx, galaxy_id, phot, err, z, pixel_posteriors, sp=sp,
                        n_samples=n_samples, 
                        pixel_type=f'{selection_label.lower().replace(" ", "_")}_pixel_{selected_pixel_idx}',
                        ax=ax, add_mags=True,
                        limits_file=limits_file,
                        path_obs_properties=path_obs_properties
                    )
                    ax2.set_yticks([27, 28, 29, 30, 31, 32, 33, 34, 35, 36])
                    ax2.minorticks_off()
                    ax2.set_xlim(4e3/1e4, 5.5e4/1e4)  
                    ax.set_xlim(4e3/1e4, 5.5e4/1e4)  
                
                except Exception as e:
                    print(f"Error plotting {galaxy_id}/{pixel_selection}: {e}")
                    ax.text(0.5, 0.5, f'Plot Error:\n{str(e)}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    continue
                
                # Set labels and titles
                if j == 0:
                    ax.set_ylabel(f'{galaxy_id}\nFlux [μJy]', fontsize=14)
                
                if i == 0:
                    ax.set_title(f"{selection_label}\n", fontsize=16)
                
                # Add pixel info as text
                info_text = f"Galaxy Pixel {selected_pixel_idx}\nFitted Index {fitted_idx}\nS/N = {snr_val:.2f}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                if i == len(galaxy_ids) - 1:
                    ax.set_xlabel("Observed wavelength [μm]")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='tab:blue', lw=1.5, label='Posterior samples', alpha=0.3),
            Line2D([0], [0], color='tab:orange', lw=1.5, label='Best fit'),
            Line2D([0], [0], color='m', lw=1.5, label='Median'),
            Line2D([0], [0], marker='.', label='Observed photometry',
                   markerfacecolor='k', markeredgecolor='k', linestyle='None', markersize=8),
            Line2D([0], [0], marker='v', label='Background limit',
                   markerfacecolor='k', markeredgecolor='k', linestyle='None', markersize=10)
        ]
        
        axes[0, 0].legend(handles=legend_elements, loc='lower right', fontsize=12)
        
        plt.tight_layout()
        
        if save_fig:
            selection_str = '_'.join([str(s) for s in pixel_selections])
            output_file = os.path.join(output_dir, f'posterior_check_{selection_str}.pdf')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {output_file}")
        
        plt.show()

# Example usage
# Run posterior predictive check (the first samples of the posteriors take a bit of time)
if __name__ == "__main__":
    from sbipix import sbipix
    # Initialize sbipix instance
    sx= sbipix()
    sx.parametric = True
    sx.both_masses = True
    # Example usage with max SNR and random pixels
    #posterior_predictive_check_sbipix(sx,data_file="obs/six_galaxies_data.hdf5",
    #                                   pixel_selections=['max_snr', 'random'],
    #                                   n_samples=500, save_fig=True)
    
    # Example with specific pixel indices
    #posterior_predictive_check_sbipix(sx,data_file="obs/six_galaxies_data.hdf5",
    #                                   pixel_selections=[10, 25],
    #                                   n_samples=500, save_fig=True)
    
    # Example with mixed selection
    posterior_predictive_check_sbipix(sx,data_file="obs/six_galaxies_data.hdf5",galaxy_ids=[206146],pixel_selections=['max_snr', 'random',5],
    n_samples=5,save_fig=False)