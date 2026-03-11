#!/usr/bin/env python3
"""
Galaxy Maps Visualization for SBIPIX

This script creates spatial maps of galaxy properties from SBIPIX inference results.
Works with both parametric and non-parametric models.

Usage:
    python galaxy_maps_sbipix.py
"""

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import os
import h5py
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import matplotlib.ticker as ticker

from sbipix import *

# Initialize SBIPIX
sx = sbipix()
sx.both_masses = True

# Cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Set matplotlib parameters for nice plots
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'medium'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['font.size'] = 20
mpl.rcParams["figure.figsize"] = (6, 5)
mpl.rcParams["mathtext.fontset"] = 'dejavuserif'

def galaxy_maps(sx, galaxy_id, data="six_galaxies_data.hdf5", 
                metric='mean', size=200, plot='phot', cmap='viridis', 
                savefig=False, all_filters=False, plot_fraction=False,
                psf_fwhm=5.36, output_dir='./galaxy_maps'):
    """
    Generate and plot galaxy maps based on SBIPIX inference results.

    Parameters:
    -----------
    sx : sbipix instance
        SBIPIX instance
    galaxy_id : int
        Galaxy ID to plot
    data : str
        HDF5 file path
    metric : str
        Metric to use for posteriors ('mean', 'median', 'mode', 'std')
    size : int
        Size of the plot region (pixels from center)
    plot : str
        Type of plot ('phot', 'err', 'sn', 'maps', 'error_maps')
    cmap : str
        Colormap for the plots
    savefig : bool
        Whether to save figures
    all_filters : bool
        Whether to plot all filters (True) or just F444W (False)
    plot_fraction : bool
        Whether to plot the fraction of fitted pixels
    psf_fwhm : float
        PSF FWHM in pixels for PSF circle
    output_dir : str
        Directory to save figures
    """

    print(f'Processing Galaxy ID: {galaxy_id}')

    # Create output directory if saving figures
    if savefig:
        os.makedirs(f'{output_dir}/{galaxy_id}', exist_ok=True)

    with h5py.File(data, "r") as hdf:
        
        # Get metadata
        filters = [f.decode('utf-8') for f in hdf["metadata"]["filters"]]
        grid_shape = hdf["metadata"].attrs["coordinates grid"]
        print(f"Filters: {len(filters)} available")
        print(f"Grid shape: {grid_shape}")
        
        # Get galaxy data
        galaxy_id_str = str(galaxy_id)
        galaxy = hdf[f"galaxies/{galaxy_id_str}"]
        z = galaxy.attrs['redshift']
        
        print(f"Redshift: {z:.2f}")

        # Initialize arrays with NaNs
        n_filters = len(filters)
        phot_gal = np.full((n_filters, grid_shape[0], grid_shape[1]), np.nan)
        err_gal = np.full((n_filters, grid_shape[0], grid_shape[1]), np.nan)

        # Fill photometry and error arrays
        coordinates = galaxy["coordinates"][:]
        photometry = galaxy["photometry"][:]
        errors = galaxy["error"][:]
        
        for k in range(len(coordinates)):
            x, y = coordinates[k]
            phot_gal[:, x, y] = photometry[k]
            err_gal[:, x, y] = errors[k]

        # Check for posteriors - handle both parametric and non-parametric
        posterior_groups = ['posterior_tau', 'posterior_nonparametric', 'posterior_dirichlet']
        posteriors = None
        max_sn = None
        labels = None
        
        for group_name in posterior_groups:
            if group_name in galaxy:
                print(f"Found posteriors: {group_name}")
                posterior_group = galaxy[group_name]
                
                if 'posteriors' in posterior_group:
                    posteriors = posterior_group['posteriors'][:]
                    max_sn = posterior_group['pixels_fitted'][:]
                    
                    # Get parameter labels
                    if 'theta' in posterior_group['posteriors'].attrs:
                        labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) 
                                for label in posterior_group['posteriors'].attrs['theta']]
                    else:
                        # Default labels based on model type
                        if 'tau' in group_name:
                            labels = ['log(M*/M☉)', 'log(M*formed/M☉)', 'log(SFR)', 'τ [Gyr]', 'ti [Gyr]', '[M/H]', 'Av']
                        else:
                            labels = ['log(M*/M☉)', 'Age [Gyr]', '[M/H]', 'Av']
                    
                    print(f"Posterior shape: {posteriors.shape}")
                    print(f"Parameters: {labels}")
                    break
        
        if posteriors is None:
            print("No posteriors found in galaxy data!")
            print("Available groups:", list(galaxy.keys()))
            if plot != 'maps':
                print("Continuing with photometry/error plots only...")
            else:
                return

        # Find center based on F444W filter (index 11)
        f444w_idx = 11  # F444W is typically index 11
        max_index = np.unravel_index(np.nanargmax(phot_gal[f444w_idx, :, :]), 
                                   phot_gal[f444w_idx].shape)
        
        # Define plot ranges
        ranges = [max_index[0] - size, max_index[0] + size, 
                 max_index[1] - size, max_index[1] + size]
        
        # Ensure ranges are within bounds
        ranges[0] = max(0, ranges[0])
        ranges[1] = min(grid_shape[0], ranges[1])
        ranges[2] = max(0, ranges[2])
        ranges[3] = min(grid_shape[1], ranges[3])

        # Plot based on type
        if plot == 'phot':
            plot_photometry(phot_gal, filters, ranges, cmap, all_filters, savefig, 
                          galaxy_id, output_dir)
            
        elif plot == 'err':
            plot_errors(err_gal, filters, ranges, cmap, all_filters, savefig, 
                       galaxy_id, output_dir)
            
        elif plot == 'sn':
            plot_signal_to_noise(phot_gal, err_gal, filters, ranges, cmap, 
                                all_filters, savefig, galaxy_id, output_dir)
            
        elif plot == 'maps' and posteriors is not None:
            plot_parameter_maps(posteriors, max_sn, coordinates, labels, 
                              phot_gal, grid_shape, ranges, metric, cmap, 
                              savefig, galaxy_id, output_dir, plot_fraction, 
                              psf_fwhm, size, sx, filters, f444w_idx)
        elif plot == 'error_maps' and posteriors is not None:
            plot_error_maps(posteriors, max_sn, coordinates, labels, 
                        grid_shape, ranges, cmap, savefig, galaxy_id, 
                        output_dir, sx, filters, f444w_idx)
    

def plot_photometry(phot_gal, filters, ranges, cmap, all_filters, savefig, 
                   galaxy_id, output_dir):
    """Plot photometry maps"""
    
    filter_indices = range(len(filters)) if all_filters else [11]  # F444W only
    
    for i in filter_indices:
        fig, ax = plt.subplots(figsize=(6, 8))
        
        data_slice = phot_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        valid_data = data_slice[np.isfinite(data_slice)]
        
        if len(valid_data) > 0:
            im = ax.imshow(data_slice, origin='lower', norm='log', cmap=cmap)
            
            filter_name = filters[i].split('_')[-1] if '_' in filters[i] else filters[i]
            cbar = plt.colorbar(mappable=im, label=f'Flux in {filter_name} [μJy]', 
                              location='top', ax=ax)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(f'Galaxy {galaxy_id} - {filter_name}', fontsize=16)
        
        if savefig:
            plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_{filter_name}_phot.pdf', 
                       bbox_inches='tight')
        plt.show()

def plot_errors(err_gal, filters, ranges, cmap, all_filters, savefig, 
               galaxy_id, output_dir):
    """Plot error maps"""
    
    filter_indices = range(len(filters)) if all_filters else [11]
    
    for i in filter_indices:
        fig, ax = plt.subplots(figsize=(6, 8))
        
        data_slice = err_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        valid_data = data_slice[np.isfinite(data_slice)]
        
        if len(valid_data) > 0:
            im = ax.imshow(data_slice, origin='lower', norm='log', cmap=cmap)
            
            filter_name = filters[i].split('_')[-1] if '_' in filters[i] else filters[i]
            cbar = plt.colorbar(mappable=im, label=f'Error in {filter_name} [μJy]', 
                              location='top', ax=ax)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(f'Galaxy {galaxy_id} - {filter_name} Error', fontsize=16)
        
        if savefig:
            plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_{filter_name}_err.pdf', 
                       bbox_inches='tight')
        plt.show()

def plot_signal_to_noise(phot_gal, err_gal, filters, ranges, cmap, all_filters, 
                        savefig, galaxy_id, output_dir):
    """Plot signal-to-noise maps"""
    
    filter_indices = range(len(filters)) if all_filters else [11]
    
    for i in filter_indices:
        fig, ax = plt.subplots(figsize=(6, 8))
        
        phot_slice = phot_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        err_slice = err_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        sn_slice = phot_slice / err_slice
        
        valid_data = sn_slice[np.isfinite(sn_slice)]
        
        if len(valid_data) > 0:
            im = ax.imshow(sn_slice, origin='lower', cmap=cmap)
            
            filter_name = filters[i].split('_')[-1] if '_' in filters[i] else filters[i]
            cbar = plt.colorbar(mappable=im, label=f'S/N in {filter_name}', 
                              location='top', ax=ax)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(f'Galaxy {galaxy_id} - {filter_name} S/N', fontsize=16)
        
        if savefig:
            plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_{filter_name}_sn.pdf', 
                       bbox_inches='tight')
        plt.show()

def plot_parameter_maps(posteriors, max_sn, coordinates, labels, phot_gal, 
                       grid_shape, ranges, metric, cmap, savefig, galaxy_id, 
                       output_dir, plot_fraction, psf_fwhm, size, sx, filters, 
                       f444w_idx):
    """Plot parameter maps from posteriors"""
    
    n_theta = len(labels)
    theta_gal = np.full((n_theta, grid_shape[0], grid_shape[1]), np.nan)
    
    # Fill parameter maps
    for k in range(len(posteriors)):
        if k < len(max_sn):
            pixel_idx = max_sn[k]
            if pixel_idx < len(coordinates):
                x, y = coordinates[pixel_idx]
                
                p = posteriors[k, :, :]
                
                if metric == 'mean':
                    t = np.mean(p, axis=0)
                elif metric == 'median':
                    t = np.median(p, axis=0)
                elif metric == 'mode':
                    t = stats.mode(np.around(p, 1), axis=0, keepdims=False)[0]
                elif metric == 'std':
                    t = np.std(p, axis=0)
                
                theta_gal[:, x, y] = t
    
    # Calculate fraction of fitted pixels
    total_pixels = len(coordinates)
    fitted_pixels = len(max_sn)
    ratio = fitted_pixels / total_pixels if total_pixels > 0 else 0
    print(f'Fitted pixels: {fitted_pixels}/{total_pixels} ({ratio:.2%})')
    
    # Set NaN for photometry where no parameters fitted
    nan_mask = np.isnan(theta_gal[0, :, :])
    phot_gal[:, nan_mask] = np.nan
    
    # Plot F444W with overlays first
    fig, ax = plt.subplots(figsize=(6, 8))
    phot_slice = phot_gal[f444w_idx, ranges[0]:ranges[1], ranges[2]:ranges[3]]
    
    if np.any(np.isfinite(phot_slice)):
        im = ax.imshow(phot_slice, origin='lower', norm='log', cmap=cmap)
        filter_name = filters[f444w_idx].split('_')[-1] if '_' in filters[f444w_idx] else filters[f444w_idx]
        cbar = plt.colorbar(mappable=im, label=f'Flux in {filter_name} [μJy]', 
                          location='top', ax=ax)
        cbar.ax.tick_params(labelsize=16)
    
    # Add PSF circle if requested
    if psf_fwhm is not None:
        center_x, center_y = size//4, size//4  # Bottom left corner
        circle = patches.Circle((center_x, center_y), radius=psf_fwhm/2, 
                              edgecolor='black', facecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(center_x, center_y - psf_fwhm, 'PSF', ha='center', va='top', 
               fontsize=14, color='black')
    
    # Add fitted pixel fraction if requested
    if plot_fraction:
        ax.text(0.05, 0.95, f'Fitted: {ratio:.1%}', fontsize=16, 
               transform=ax.transAxes, color='white', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title(f'Galaxy {galaxy_id} - {filter_name}', fontsize=16)
    
    if savefig:
        plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_{filter_name}_reference.pdf', 
                   bbox_inches='tight')
    plt.show()
    
    # Plot parameter maps
    for i in range(n_theta):
        fig, ax = plt.subplots(figsize=(6, 8))
        
        param_slice = theta_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        
        # Handle reversed colormaps for certain parameters
        if sx.both_masses:
            reverse_params = [2, 3, 4]  # SFR, age-related parameters
        else:
            reverse_params = [1, 2, 3]
        
        cmap_use = cmap + '_r' if i in reverse_params else cmap
        
        if np.any(np.isfinite(param_slice)):
            im = ax.imshow(param_slice, origin='lower', cmap=cmap_use)
            
            # Add total stellar mass annotation for mass parameter
            if i == 0 and 'log(M' in labels[i]:
                total_mass = np.log10(np.nansum(10 ** theta_gal[i, :, :]))
                ax.text(0.05, 0.05, f'log(M*,tot/M☉) = {total_mass:.2f}', 
                       fontsize=16, transform=ax.transAxes, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            cbar = plt.colorbar(mappable=im, label=labels[i], location='top', ax=ax)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(f'Galaxy {galaxy_id} - {labels[i]}', fontsize=16)
        
        if savefig:
            plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_param_{i}_{labels[i].replace("/", "_")}.pdf', 
                       bbox_inches='tight')
        plt.show()

def plot_error_maps(posteriors, max_sn, coordinates, labels, grid_shape, 
                   ranges, cmap, savefig, galaxy_id, output_dir, sx, 
                   filters, f444w_idx):
    """Plot error maps (standard deviation) from posteriors"""
    
    n_theta = len(labels)
    theta_std_gal = np.full((n_theta, grid_shape[0], grid_shape[1]), np.nan)
    
    # Fill error maps with standard deviations
    for k in range(len(posteriors)):
        if k < len(max_sn):
            pixel_idx = max_sn[k]
            if pixel_idx < len(coordinates):
                x, y = coordinates[pixel_idx]
                
                p = posteriors[k, :, :]
                t_std = np.std(p, axis=0)  # Standard deviation for each parameter
                
                theta_std_gal[:, x, y] = t_std
    
    # Calculate fraction of fitted pixels
    total_pixels = len(coordinates)
    fitted_pixels = len(max_sn)
    ratio = fitted_pixels / total_pixels if total_pixels > 0 else 0
    print(f'Error maps - Fitted pixels: {fitted_pixels}/{total_pixels} ({ratio:.2%})')
    
    # Plot error maps for each parameter
    for i in range(n_theta):
        fig, ax = plt.subplots(figsize=(6, 8))
        
        error_slice = theta_std_gal[i, ranges[0]:ranges[1], ranges[2]:ranges[3]]
        
        if np.any(np.isfinite(error_slice)):
            # Use linear scale for error maps (not log)
            im = ax.imshow(error_slice, origin='lower', cmap=cmap)
            
            # Add statistics to the plot
            valid_errors = error_slice[np.isfinite(error_slice)]
            if len(valid_errors) > 0:
                median_error = np.median(valid_errors)
                ax.text(0.05, 0.95, f'Med. σ = {median_error:.3f}', 
                       fontsize=16, transform=ax.transAxes, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            cbar = plt.colorbar(mappable=im, label=f'σ({labels[i]})', location='top', ax=ax)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        if savefig:
            plt.savefig(f'{output_dir}/{galaxy_id}/ID_{galaxy_id}_error_{i}_{labels[i].replace("/", "_")}.pdf', 
                       bbox_inches='tight')
        plt.show()

def run_all_galaxies(data_file="six_galaxies_data.hdf5", 
                    plot_type='maps', sizes=None):
    """
    Run galaxy maps for all galaxies in the HDF5 file
    
    Parameters:
    -----------
    data_file : str
        Path to HDF5 file
    plot_type : str
        Type of plots to generate
    sizes : list or None
        Custom sizes for each galaxy, or None for automatic sizing
    """
    
    # Default galaxy list and sizes
    galaxy_ids = [254985, 205449, 211273, 117960, 118081, 206146]
    
    if sizes is None:
        sizes = [100, 75, 60, 50, 50, 50]  # Default sizes
    
    print(f"Processing {len(galaxy_ids)} galaxies...")
    print(f"Plot type: {plot_type}")
    print("=" * 50)
    
    for k, galaxy_id in enumerate(galaxy_ids):
        size = sizes[k] if k < len(sizes) else 100
        
        print(f"\nProcessing galaxy {k+1}/{len(galaxy_ids)}: {galaxy_id}")
        print(f"Using size: {size}")
        
        try:
            galaxy_maps(
                sx=sx,
                galaxy_id=galaxy_id,
                data=data_file,
                size=size,
                plot=plot_type,
                cmap='turbo',
                savefig=False,
                all_filters=False,
                plot_fraction=True,
                metric='mean'
            )
            print(f"Successfully processed galaxy {galaxy_id}")
            
        except Exception as e:
            print(f"Error processing galaxy {galaxy_id}: {e}")
            continue
        
        print("-" * 30)
    
    print("\nFinished processing all galaxies!")

if __name__ == "__main__":
    
    print("SBIPIX Galaxy Maps Visualization")
    print("=" * 40)

    local_path = os.path.dirname(os.path.abspath(__file__))

    # Example 1: Single galaxy
    print("Example: Single galaxy maps")
    galaxy_maps(
        sx=sx,
        galaxy_id=211273,
        data=local_path+"/../obs/six_galaxies_data.hdf5",
        size=100,
        plot='maps',
        cmap='turbo',
        savefig=False,
        plot_fraction=True,
        all_filters=True
    )
    
    """# Example 2: All galaxies
    print("\n" + "="*50)
    print("Processing all galaxies...")
    run_all_galaxies(
        data_file=local_path+"/obs/six_galaxies_data.hdf5",
        plot_type='maps'
    )"""