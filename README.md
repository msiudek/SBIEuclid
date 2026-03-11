## SBIPIX: Simulation-based inference of galaxy properties from JWST pixels

[![arXiv](https://img.shields.io/badge/arXiv-2203.07391-B31B1B.svg)](https://arxiv.org/abs/2506.04336)

---

SBIPIX is a fast Bayesian SED fitting tool for inferring galaxy properties (stellar mass, star formation history, metallicity, dust) from JWST imaging. It uses simulation-based inference to analyze individual galaxy pixels, taking advantage of JWST's high spatial resolution and multiwavelength coverage.

## Getting Started

### Prerequisites

SBIPIX requires Python 3.8+


### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/patriglesias/SBIPIX.git
cd SBIPIX
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Verify installation:**
```python
python -c "from sbipix import sbipix; print('SBIPIX installed successfully!')"
```

### Quick Start

#### 1. Initialize SBIPIX
```python
from sbipix import sbipix
from sbipix import *
from sbipix.plotting.diagnostics  import plot_test_performance

# Create SBIPIX instance
sx = sbipix()

# Configure for JADES
sx.filter_path = 'obs/obs_properties/'
sx.filter_list='filters_jades_no_wfc.dat'
sx.atlas_name = 'atlas_obs_jades'
sx.n_simulation = 1000
sx.parametric = True  # or False for non-parametric SFH
sx.both_masses = True
sx.infer_z = False
```

#### 2. Run a Basic Simulation
```python
# Simulate galaxy SEDs
sx.simulate(
    mass_max=12, mass_min=4.0,
    z_prior='flat', z_min=0.0, z_max=7.5,
    Z_min=-2.27, Z_max=0.4,
    dust_model='Calzetti',
    Av_min=0.0, Av_max=4.0
)

# Load simulation results
sx.load_simulation()
```

#### 3. Add Observational Effects
```python
# Load observational properties
sx.include_limit = True
sx.condition_sigma = True
sx.include_sigma = True
sx.load_obs_features()
sx.add_noise_nan_limit_all()
```

#### 4. Train the Model
```python
# Train normalizing flow
sx.train(
    min_thetas=min_values,
    max_thetas=max_values,
    n_max=len(sx.theta),
    nblocks=5,
    nhidden=128
)
```

#### 5. Test Performance
```python
# Test model performance
posterior_test = sx.test_performance(n_test=100, return_posterior=True)
sx.plot_test_performance(sx,n_test=100)

```

### Working with JADES Data

#### Inspect Sample Galaxies
```python
from examples.inspect_jades_hdf5 import JADESInspector

# Initialize inspector
inspector = JADESInspector("obs/six_galaxies_data.hdf5")

# Overview of the dataset
inspector.print_overview()

# Inspect specific galaxy
inspector.inspect_galaxy("205449")

# Plot galaxy footprint
inspector.plot_galaxy_footprint("205449")

# Plot SED (median, specific pixel, or max S/N pixel)
inspector.plot_sed("205449", pixel_idx='max_snr', show_upper_limits=True)
```

#### Run Inference on Sample Data
```python
from examples.inference_six_gal import galaxy_inference, run_inference_for_all_galaxies

# Single galaxy inference
posteriors = galaxy_inference(
    sx, 
    galaxy_id=205449,
    data="obs/six_galaxies_data.hdf5",
    sn_limit=5.0,
    n_samples=500,
    save_posteriors=True
)

# Process all sample galaxies
run_inference_for_all_galaxies(
    hdf5_file="obs/six_galaxies_data.hdf5",
    sn_limits=[5, 5, 5, 5, 5, 5]
)
```

#### Visualize Results
```python
from examples.galaxy_maps_sbipix import galaxy_maps, run_all_galaxies

# Create parameter maps for a single galaxy
galaxy_maps(
    sx=sx,
    galaxy_id=205449,
    size=75,
    plot='maps',  # 'phot', 'err', 'sn', or 'maps'
    cmap='turbo',
    plot_fraction=True
)

# Process all galaxies
run_all_galaxies(
    data_file="obs/six_galaxies_data.hdf5",
    plot_type='maps'
)
```

### Repository Structure

```
SBIPIX/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── src/sbipix/               # Main package
│   ├── __init__.py       
│   ├── sbipix.py             # Core SBIPIX class
│   ├── plotting/             # Plotting utilities
│   │   ├── __init__.py
│   │   └── diagnostics.py
│   ├── train/                # Training modules
│   │   ├── __init__.py
│   │   └── simulator.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── cosmology.py
│       └── sed_utils.py
├── examples/                 # Example scripts and workflows
│   ├── getting_started.ipynb # Getting started tutorial notebook
│   ├── galaxy_maps.py        # Visualization tool
│   ├── inference_six_gal.py  # Inference pipeline
│   └── simulation_training_testing.py  # Training example
├── obs/                      # Observational data and tools
│   ├── inspect_jades_hdf5.py # Data inspection tool
│   ├── six_galaxies_data.hdf5 # Sample JADES data
│   └── obs_properties/       # Filter curves and noise models
│       ├── FILTERS_HST/      # HST filter transmission curves
│       ├── FILTERS_JWST/     # JWST filter transmission curves
│       ├── filters_jades_no_wfc.dat
│       └── *.npy             # Noise and calibration files
└── library/                  # Pre-computed stellar libraries
    ├── atlas_obs_jades_1000_Nparam_2.dbatlas
    └── atlas_obs_jades_1000_Nparam_3.dbatlas
```

### Next Steps

- **Tutorial Notebooks**: See `examples/` for Jupyter notebooks with detailed walkthroughs
- **API Documentation**: Run `help(sbipix)` for full API reference  
- **Sample Data**: Download JADES sample data from [link]
- **Pre-trained Models**: Access trained models at [link]

### Citation

If you use SBIPIX in your research, please cite:
```bibtex
@ARTICLE{2025A&A...703A.229I,
       author = {{Iglesias-Navarro}, Patricia and {Huertas-Company}, Marc and {P{\'e}rez-Gonz{\'a}lez}, Pablo and {Knapen}, Johan H. and {Hahn}, ChangHoon and {Koekemoer}, Anton M. and {Finkelstein}, Steven L. and {Villanueva}, Natalia and {Asensio Ramos}, Andr{\'e}s},
        title = "{Simulation-based inference of galaxy properties from JWST pixels}",
      journal = {\aap},
     keywords = {galaxies: evolution, galaxies: fundamental parameters, galaxies: star formation, galaxies: statistics, Astrophysics of Galaxies},
         year = 2025,
        month = nov,
       volume = {703},
          eid = {A229},
        pages = {A229},
          doi = {10.1051/0004-6361/202555810},
archivePrefix = {arXiv},
       eprint = {2506.04336},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...703A.229I},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```