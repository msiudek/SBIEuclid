from sbipix import *
from sbipix.plotting.diagnostics  import plot_filters,plot_theta, plot_test_performance
import os

# Initialize the sbipix class
sx = sbipix()

# Configuration: survey and filters
sx.filter_path= os.path.abspath('.')+'/obs/obs_properties/' # Path to filter files and observational features
sx.filter_list = 'filters_jades_no_wfc.dat'  # JWST JADES filter set (ACS but no WFC)
sx.atlas_name = 'atlas_obs_jades'            # atlas for JADES survey

#plot the filter set
plot_filters(sx)

# Simulation parameters
sx.n_simulation = 1000      # Number of simulated galaxies
sx.parametric = False       # Use non-parametric SFH 
sx.model_name = 'post_obs_jades.pkl'  # Output file for trained model

# Run simulation with parameter ranges (first simulations take time, later is much faster!)
sx.simulate(
    # Mass range (log10 solar masses), flat prior
    mass_max = 12,           # Max: 10^12 M_sun
    mass_min = 4.0,          # Min: 10^4 M_sun
    
    # Redshift prior and range
    z_prior = 'flat',        # Uniform redshift prior
    z_min = 0.0,             # Local universe
    z_max = 7.5,             # High redshift limit
    
    # Metallicity range (log10 Z/Z_sun), flat prior
    Z_min = -2.27,           # ~1/200 solar metallicity
    Z_max = 0.4,             # ~2.5x solar metallicity
    
    # Dust attenuation
    dust_model = 'Calzetti', # Calzetti dust law
    dust_prior = 'flat',     # Uniform dust prior
    Av_min = 0.0,            # No dust
    Av_max = 4.0             # Heavy dust extinction
)

# Load the simulated data for training
sx.load_simulation()

# plot the simulated distributions of parameters
plot_theta(sx,limit_sfr=True,range_sfr=(-15,4))

"""
# Add realism to the simulation
sx.include_limit = True      # Include upper limits in training data
sx.condition_sigma = True    # Condition on photometric uncertainties in the training
sx.include_sigma = True      # add uncertainties to the simulated fluxes

# Load observational features and apply realistic conditions
sx.load_obs_features()                # Load real observation features (upper limits and noise)
sx.add_noise_nan_limit_all()         # Add observational features to the simulation

# Redshift configuration
sx.infer_z = False                    # Use redshift as input (not inferred parameter)

# Data cleaning: remove NaNs and infinite values from simulation
sim_ok = np.isfinite(np.sum(sx.theta, axis=1))  # Find finite parameter combinations
sx.theta = sx.theta[sim_ok, :]                   # Keep only valid parameters
sx.mag = sx.mag[sim_ok, :, :]                    # Keep corresponding magnitudes
sx.obs = sx.obs[sim_ok, :]                       # Keep corresponding fluxes
sx.n_simulation = len(sx.theta[:, 0])           # Update simulation count

# Calculate parameter bounds for training
max_thetas = np.max(sx.theta[:, :], axis=0)     # Maximum values per parameter
min_thetas = np.min(sx.theta[:, :], axis=0)     # Minimum values per parameter
print(min_thetas)                                # Display parameter ranges
print(max_thetas)

# Train the Normalizing Flow model
m = min_thetas                                   # Store min bounds
M = max_thetas                                   # Store max bounds
sx.train(
    min_thetas = m,              # Parameter lower bounds for training
    max_thetas = M,              # Parameter upper bounds for training
    n_max = len(sx.theta),       # Use all available training data
    nblocks = 5,                 # Number of coupling blocks in flow
    nhidden = 128                # Hidden units per layer 
)

# Model evaluation and diagnostics
posterior_test = sx.test_performance(
    n_test = 100,                # Test on 100 random samples
    return_posterior = True      # Return full posterior samples
)

# Generate performance plots
plot_test_performance(sx,n_test = 100)  # Create diagnostic plots"""