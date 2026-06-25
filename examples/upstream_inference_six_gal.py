from sbipix import *
import os
import h5py

# Initialize SBIPIX
sx = sbipix()

# Parameter labels for plotting and saving
labels = ['log($M_{*}/\\rm{M}_{\\odot}$)', 
         'log($M_{*}^{\\rm{formed}}/\\rm{M}_{\\odot}$)', 
         'log(SFR/($\\rm{M}_{\\odot}$/yr))', 
         '$\\tau$ [Gyr]', 
         '$t_i$ [Gyr]', 
         '[M/H]', 
         'Av']

# Cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

"""SBIPIX parameters"""
local_path=os.path.abspath('.')
sx.model_name = 'post_obs_jades_resolved_tau_z7.5_cpu.pkl' # Adjust this to your model, this is a model trained on cpu up to z=7.5 with tau-delayed SFHs on 100k simulations
sx.filter_path = local_path+'/obs/obs_properties/'
sx.model_path = local_path+'/library/'
sx.n_simulation = 1000000
sx.parametric = True
sx.both_masses = True
sx.infer_z = False
sx.include_limit = True
sx.condition_sigma = True
sx.include_sigma = True

def galaxy_inference(sx, id, data="six_galaxies_data.hdf5", device='cpu', sn_limit=None, 
                    n_theta=7, n_samples=None, compute_integrated_properties=False, 
                    save_posteriors=False):
    """
    Function to infer galaxy properties using SBIPIX.

    Parameters:
    -----------
    sx : sbipix instance
        Instance of the sbipix class
    id : int
        Galaxy ID
    data : str
        HDF5 file that stores photometry, errors, coordinates and redshift
    device : str
        Device to run the code ('cuda' or 'cpu')
    sn_limit : float
        Signal-to-noise limit for pixel selection
    n_theta : int
        Number of parameters
    n_samples : int or None
        Number of samples for the posteriors
    compute_integrated_properties : bool
        Boolean to compute integrated properties
    save_posteriors : bool
        Boolean to save posteriors to HDF5 file
    
    Returns:
    --------
    p : array
        Posterior samples (shape depends on compute_integrated_properties)
    """

    # Print the ID of the galaxy
    print(f'Galaxy ID: {id}')

    # Get the galaxy data by loading the hdf5 file
    with h5py.File(data, "a") as hdf:
        print("Filters:", [f.decode('utf-8') for f in hdf["metadata"]["filters"]])
        print("Units:", hdf["metadata"].attrs["units"])
        print("Coordinates grid:", hdf["metadata"].attrs["coordinates grid"])
        
        galaxy_id = str(id)
        galaxy = hdf[f"galaxies/{galaxy_id}"]
        
        # Get redshift
        redshift = galaxy.attrs.get("redshift", "N/A")
        print(f"Redshift: {redshift:.2f}" if isinstance(redshift, (int, float)) else f"Redshift: {redshift}")

        # Calculate S/N for F277W, F356W, F444W filters
        # Filter indices: F277W=6, F356W=8, F444W=11 (adjust if needed)
        filters_sn = [6, 8, 11]  # F277W, F356W, F444W
        
        print(f"Using filters for S/N calculation: {[hdf['metadata']['filters'][i].decode('utf-8') for i in filters_sn]}")
        
        s_n = np.nanmean(galaxy["photometry"][:, filters_sn] / galaxy["error"][:, filters_sn], axis=1)

        # Select pixels above S/N limit
        if sn_limit is None:
            sn_limit = 5.0  # Default S/N limit
            
        max_sn = np.where(s_n > sn_limit)[0]
        
        print(f'Minimum S/N = {np.nanmin(s_n[max_sn]):.2f}')
        print(f'Computing posteriors for {len(max_sn)} pixels (S/N > {sn_limit})')
        print(f'Total pixels in galaxy: {len(galaxy["photometry"])}')

        if len(max_sn) == 0:
            print(f"ERROR: No pixels above S/N limit {sn_limit}!")
            return None

        if n_samples is None:
            n_samples = 500

        if not compute_integrated_properties:
            # Compute resolved posteriors for individual pixels
            print("Computing resolved posteriors...")
            p = sx.get_posteriors_resolved(
                galaxy["photometry"][max_sn[:], :], 
                id, 
                input_z=galaxy.attrs["redshift"],
                n_samples=n_samples, 
                save=False, 
                return_stats=False, 
                sigma_arr=galaxy["error"][max_sn[:], :],
                device=device
            )

            if save_posteriors:
                print("Saving posteriors to HDF5...")
                # Remove existing posterior group if it exists
                if "posterior_tau" in hdf[f"galaxies/{galaxy_id}"]:
                    del hdf[f"galaxies/{galaxy_id}"]["posterior_tau"]
                
                # Create new posterior group
                posteriors = hdf[f"galaxies/{galaxy_id}"].create_group("posterior_tau")
                posterior_dataset = posteriors.create_dataset("posteriors", data=p)
                posteriors.create_dataset("pixels_fitted", data=max_sn)
                
                # Save metadata
                posterior_dataset.attrs["min_SNR"] = sn_limit
                posterior_dataset.attrs["n_samples"] = n_samples
                posterior_dataset.attrs["theta"] = [label.encode('utf-8') for label in labels]
                posterior_dataset.attrs["inference_method"] = "sbipix_resolved"
                
                print(f"Saved posteriors for {len(max_sn)} pixels")

        else:
            # Compute integrated properties
            print("Computing integrated properties...")
            phot_int = np.sum(galaxy["photometry"][max_sn[:], :], axis=0)
            err_int = np.sqrt(np.sum(galaxy["error"][max_sn[:], :]**2, axis=0))
            
            print(f'Integrated photometry: {phot_int}')
            print(f'Integrated errors: {err_int}')
            print(f'Number of pixels: {len(max_sn)}')
            
            # Expand dimensions for single "pixel" (integrated)
            phot_int = np.expand_dims(phot_int, axis=0)
            err_int = np.expand_dims(err_int, axis=0)
            
            p = sx.get_posteriors_resolved(
                np.copy(phot_int), 
                id, 
                n_samples=n_samples, 
                save=False, 
                return_stats=False, 
                sigma_arr=np.copy(err_int), 
                input_z=galaxy.attrs["redshift"],
                device=device
            )
            
            print(f'Integrated stellar mass: {np.mean(p[0,:,0]):.2f} ± {np.std(p[0,:,0]):.2f} log(M_sun)')
            
            if save_posteriors:
                # Save integrated posteriors
                posteriors = hdf[f"galaxies/{galaxy_id}/posterior_tau"]
                if "posterior_integrated" in posteriors:
                    del posteriors["posterior_integrated"]
                
                posterior_dataset = posteriors.create_dataset("posterior_integrated", data=p)
                posterior_dataset.attrs["method"] = "integrated"
                posterior_dataset.attrs["pixels_fitted"] = max_sn
                posterior_dataset.attrs["min_SNR"] = sn_limit
                
                print("Saved integrated posteriors")

    return p

def run_inference_for_all_galaxies(hdf5_file="six_galaxies_data.hdf5", sn_limits=None):
    """
    Run inference for all galaxies in the HDF5 file
    
    Parameters:
    -----------
    hdf5_file : str
        Path to HDF5 file
    sn_limits : list or None
        S/N limits for each galaxy, or None for default
    """
    
    # Galaxy IDs (update these to match your HDF5 file)
    galaxy_ids = [254985, 205449, 211273, 117960, 118081, 206146]
    
    # Default S/N limits if not provided
    if sn_limits is None:
        sn_limits = [5, 5, 5, 5, 5, 5]  # Same S/N limit for all galaxies
    
    print("Starting inference for all galaxies...")
    print(f"Using HDF5 file: {hdf5_file}")
    print(f"Galaxy IDs: {galaxy_ids}")
    print(f"S/N limits: {sn_limits}")
    print("=" * 60)
    
    for k, galaxy_id in enumerate(galaxy_ids):
        print(f'\nProcessing Galaxy {k+1}/{len(galaxy_ids)}: {galaxy_id}')
        print("-" * 40)
        
        try:
            p = galaxy_inference(
                sx, 
                galaxy_id, 
                data=hdf5_file,
                device='cpu',
                sn_limit=sn_limits[k],
                n_theta=7,
                n_samples=500,  # You can adjust this
                compute_integrated_properties=False,
                save_posteriors=True
            )
            
            if p is not None:
                print(f"Successfully processed galaxy {galaxy_id}")
                print(f"Posterior shape: {p.shape}")
            else:
                print(f"Failed to process galaxy {galaxy_id}")
                
        except Exception as e:
            print(f"ERROR processing galaxy {galaxy_id}: {e}")
        
        print("-" * 40)
    
    print("\nFinished processing all galaxies!")

# Example usage
if __name__ == "__main__":
    # Load the obs features
    sx.load_obs_features()

    # Option 1: Run inference for a single galaxy
    print("Example: Single galaxy inference")
    p = galaxy_inference(
        sx, 
        id=211273, 
        data=local_path+"/obs/six_galaxies_data.hdf5",
        device='cpu',
        sn_limit=5.0,
        n_samples=500,
        save_posteriors=True,
        compute_integrated_properties=False
    )
    
    """# Option 2: Run inference for all galaxies
    print("\n" + "="*60)
    print("Running inference for all galaxies...")
    run_inference_for_all_galaxies(
        hdf5_file=local_path+"/obs/six_galaxies_data.hdf5",
        sn_limits=[5, 5, 5, 5, 5, 5]
    )"""