"""
SBIPIX: Main class for simulation-based inference on pixel-level stellar population properties
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sbi
from sbi import utils as Ut
from sbi import inference as Inference
import pickle
import seaborn as sns
import sklearn.metrics as sm
from scipy import stats
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from tqdm import tqdm, trange
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from .utils.sed_utils import mag_conversion
from .utils.cosmology import setup_cosmology
from .train.simulator import generate_atlas_parametric
from .plotting.diagnostics import plot_test_performance


DEBUG_SIGMA_ONLY = os.getenv("SBIPIX_DEBUG_SIGMA_ONLY", "0") == "1"
DEBUG_NOISE_ONLY = os.getenv("SBIPIX_DEBUG_NOISE_ONLY", "0") == "1"
DEBUG_MASK_DIAGNOSTICS = os.getenv("SBIPIX_DEBUG_MASK_DIAGNOSTICS", "0") == "1"


class sbipix():
    """
    A class for simulation-based inference pipeline for studying stellar population 
    properties on integrated/resolved galaxies from JWST.

    This class provides a complete workflow for:
    1. Simulating galaxy SEDs with various star formation histories
    2. Training neural density estimators using simulation-based inference
    3. Inferring stellar population properties from observed photometry

    Attributes
    ----------
    n_filters : int
        Number of filters used from the filter_list (default: 19)
    filter_list : str
        Text file with paths for the filter files
    filter_path : str
        Path where filter_list is located
    atlas_path : str
        Path where atlas is located
    atlas_name : str
        Name of atlas object
    n_simulation : int
        Number of simulated galaxies for training
    parametric : bool
        If True, use parametric (τ-delayed) SFH; if False, use Dirichlet prior
    both_masses : bool
        If True, include both formed and surviving stellar masses
    infer_z : bool
        If True, infer redshift from photometry
    obs : np.ndarray
        Array of shape (n_simulation, n_filters) with simulated photometry
    theta : np.ndarray
        Array of shape (n_simulation, n_params) with physical properties
    labels : list
        Names of the physical properties in theta
    mag : np.ndarray
        Processed magnitudes ready for training (with noise, masks, limits)
    
    Observational Properties
    -----------------------
    include_sigma : bool
        Include photometric uncertainties in simulation
    include_mask : bool
        Include masking for unavailable filters
    include_limit : bool
        Include detection limits
    condition_sigma : bool
        Include uncertainties as network input
    mean_sigma_obs : np.ndarray
        Mean uncertainty distributions per magnitude bin and filter
    stds_sigma_obs : np.ndarray
        Standard deviation of uncertainty distributions
    percentiles : np.ndarray
        Percentiles for magnitude bins used to assign uncertainties
    limits : np.ndarray
        1σ depth limits for each filter
    
    Model Properties
    ---------------
    model_path : str
        Path for saving/loading trained models
    model_name : str
        Filename for the trained model
    means_test : np.ndarray
        Test set posterior means
    stds_test : np.ndarray
        Test set posterior standard deviations
    
    Observational Data
    -----------------
    catalog_path : str
        Path to observational catalogs
    catalog_name : str
        Name of the observational catalog
    mag_obs : np.ndarray
        Processed observational photometry
    posteriors_obs : np.ndarray
        Inferred posteriors for observed galaxies
    means_obs : np.ndarray
        Posterior means for observed galaxies
    stds_obs : np.ndarray
        Posterior uncertainties for observed galaxies

    Examples
    --------
    >>> # Basic usage for parametric SFH
    >>> model = SBIPIX()
    >>> model.parametric = True
    >>> model.simulate(n_simulation=50000)
    >>> model.load_obs_features()
    >>> model.add_noise_nan_limit_all()
    >>> model.train()
    >>> model.test_performance()
    
    >>> # For resolved galaxy analysis
    >>> posteriors = model.get_posteriors_resolved(phot_data, n_gal=10)
    """

    def __init__(self):
        """Initialize SBIPIX with default configuration for JADES analysis."""
        # Filter and data configuration
        self.n_filters = 19
        self.filter_list = 'filters_jades_no_wfc.dat'
        self.filter_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'obs', 'obs_properties')
        )
        self.mean_sigma_file = 'mean_sigma_jades_res_bins.npy'
        self.std_sigma_file = 'std_sigma_jades_res_bins.npy'
        self.sigma_samples_file = None
        self.percentiles_file = 'percentiles_jades_res_bins.npy'
        self.limits_file = 'background_noise_hainline.npy'
        self.lam_eff_file = 'lam_eff.npy'
        self.atlas_name = 'atlas_obs_jades_no_wfc'
        self.atlas_path = './library/'
        
        # Simulation parameters
        self.n_simulation = 100000
        self.parametric = False  # Use Dirichlet by default
        self.both_masses = False
        self.remove_filters = None
        
        # Data arrays (initialized as None)
        self.obs = None
        self.theta = None
        self.mag = None
        
        # Parameter labels (updated based on SFH type)
        self.labels = [
            'log($\\rm{M}_{*}/\\rm{M}_{\\odot}$)',
            'log(SFR/($\\rm{M}_{\\odot}$/yr))',
            '$t_{25\\%}$', '$t_{50\\%}$', '$t_{75\\%}$',
            '[M/H]', 'Av', 'z'
        ]
        
        # Observational realism parameters
        self.include_sigma = False
        self.include_mask = False
        self.include_limit = False
        self.condition_sigma = False
        self.noise_sigma_sampler = 'empirical'
        self.noise_detection_model = 'hard'
        self.noise_observation_space = 'mag'
        self.noise_sigma_mag_params = None
        self.noise_sigma_mag_ranges = None
        self.noise_sigma_mag_interp_centers = None
        self.noise_sigma_mag_interp_means = None
        self.noise_sigma_mag_interp_stds = None
        # Valid domain per filter — only interpolate within this range
        self.noise_sigma_mag_valid_min = None   # (n_filters,) faintest trained bin center
        self.noise_sigma_mag_valid_max = None   # (n_filters,) brightest trained bin center
        self.noise_sigma_flux_alpha = None
        self.noise_sigma_floor = 8e-3
        self.noise_sigma_mag_min = 1e-6
        self.noise_sigma_mag_max = 10.0
        self.snr_threshold = 1.0
        
        # Observational properties (loaded from files)
        self.mean_sigma_obs = None
        self.stds_sigma_obs = None
        self.sigma_samples_obs = None
        self.percentiles = None
        self.limits = None
        
        # Model configuration
        self.model_path = "./library/"
        self.model_name = "posteriors.pkl"
        self.infer_z = True
        self.infer_z_integrated = False
        
        # Results storage
        self.means_test = None
        self.stds_test = None
        self.mode_test = None
        
        # Observational data
        self.catalog_path = './JADES/'
        self.catalog_name = 'ra_dec_mach_phot_spec_z.fits'
        self.mag_obs = None
        self.flags_obs = None
        self.id_specz = None
        self.id_photoz = None
        self.posteriors_obs = None
        self.means_obs = None
        self.stds_obs = None
        self.mode_obs = None
        self.ind_obs = None
        self.gal = None
        
        # Analysis type
        self.type = 'Resolved'  # 'Integrated' or 'Resolved'

        self.refresh_filter_metadata()

    def _resolve_filter_list_path(self):
        """Resolve absolute path to the filter list file."""
        if os.path.isabs(self.filter_list):
            return self.filter_list
        return os.path.join(self.filter_path, self.filter_list)

    def refresh_filter_metadata(self):
        """Refresh filter metadata (currently infers number of filters from list file)."""
        filter_list_path = self._resolve_filter_list_path()
        try:
            with open(filter_list_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
            valid_lines = [line for line in lines if line and not line.startswith('#')]
            self.n_filters = len(valid_lines)
        except OSError:
            pass

    def _prepare_dense_basis_filter_list(self):
        """
        Build a dense_basis-compatible one-column filter-list file.

        Our project filter list may be multi-column (e.g., path, short name,
        col_stem). dense_basis expects a plain one-column list of filter paths.
        This method writes a temporary one-column file in self.filter_path and
        returns its filename (basename), suitable for dense_basis fkit_name.
        """
        src_path = self._resolve_filter_list_path()
        out_name = "_dense_basis_filter_list.dat"
        out_path = os.path.join(self.filter_path, out_name)

        with open(src_path, 'r', encoding='utf-8') as src:
            lines = src.readlines()

        one_col = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            # First token is always the relative filter transmission path.
            one_col.append(stripped.split()[0])

        with open(out_path, 'w', encoding='utf-8') as out:
            out.write("\n".join(one_col) + "\n")

        return out_name

    def configure_filters(self, filter_list=None, filter_path=None,
                          mean_sigma_file=None, std_sigma_file=None,
                          sigma_samples_file=None,
                          percentiles_file=None, limits_file=None,
                          lam_eff_file=None):
        """Configure filter-related files in one place and refresh metadata."""
        if filter_list is not None:
            self.filter_list = filter_list
        if filter_path is not None:
            self.filter_path = filter_path
        if mean_sigma_file is not None:
            self.mean_sigma_file = mean_sigma_file
        if std_sigma_file is not None:
            self.std_sigma_file = std_sigma_file
        if sigma_samples_file is not None:
            self.sigma_samples_file = sigma_samples_file
        if percentiles_file is not None:
            self.percentiles_file = percentiles_file
        if limits_file is not None:
            self.limits_file = limits_file
        if lam_eff_file is not None:
            self.lam_eff_file = lam_eff_file

        self.refresh_filter_metadata()

    def configure_noise_model(self,
                              sigma_sampler=None,
                              detection_model=None,
                              observation_space=None):
        """Configure how observational uncertainties are sampled."""
        if sigma_sampler is not None:
            self.noise_sigma_sampler = str(sigma_sampler)
        if detection_model is not None:
            self.noise_detection_model = str(detection_model)
        if observation_space is not None:
            value = str(observation_space).strip().lower()
            if value not in ('mag', 'flux'):
                raise ValueError(
                    f"Unsupported observation_space={observation_space!r}. "
                    "Use 'mag' or 'flux'."
                )
            self.noise_observation_space = value

    def _compute_bin_centers(self, filter_idx):
        """Return magnitude-bin centers from percentile boundaries."""
        percentiles_f = np.asarray(self.percentiles[:, filter_idx], dtype=float)
        means_f = np.asarray(self.mean_sigma_obs[filter_idx], dtype=float)

        if len(percentiles_f) == 0 or not np.isfinite(percentiles_f).all():
            centers = np.arange(len(means_f), dtype=float) + 20.0
            return centers, means_f

        if len(percentiles_f) == 1:
            centers = np.array([percentiles_f[0] - 0.5, percentiles_f[0] + 0.5])
        else:
            left_center = percentiles_f[0] - 0.5 * (percentiles_f[1] - percentiles_f[0])
            right_center = percentiles_f[-1] + 0.5 * (percentiles_f[-1] - percentiles_f[-2])
            centers = np.concatenate([
                [left_center],
                0.5 * (percentiles_f[:-1] + percentiles_f[1:]),
                [right_center],
            ])
        return centers, means_f

    def _sample_sigma_distribution(self, pixel_means, pixel_stds):
        """Sample per-object uncertainty."""
        pixel_means = np.maximum(np.asarray(pixel_means, dtype=float), 1e-6)
        pixel_stds = np.maximum(np.asarray(pixel_stds, dtype=float), 1e-6)

        # Keep a single legacy fallback sampler here (truncated normal).
        # The dedicated 'mag_lognormal' sampler is handled explicitly in
        # _add_noise_nan_limit via _sample_sigma_mag_lognormal().
        a_std = (0.0 - pixel_means) / pixel_stds
        mag_errs_det = truncnorm.rvs(
            a=a_std,
            b=np.inf,
            loc=pixel_means,
            scale=pixel_stds
        )

        return mag_errs_det

    def _fit_sigma_mag_lognormal_params(self, filter_idx):
        """Fit log(sigma) = a + b*mag + c*mag^2 and scatter for one filter."""
        centers, means_f = self._compute_bin_centers(filter_idx)
        means_f = np.asarray(means_f, dtype=float)
        stds_f = np.asarray(self.stds_sigma_obs[filter_idx], dtype=float)

        valid = np.isfinite(centers) & np.isfinite(means_f) & (means_f > 0)
        if np.sum(valid) < 3:
            mean_fallback = np.nanmedian(means_f[np.isfinite(means_f) & (means_f > 0)])
            if not np.isfinite(mean_fallback) or mean_fallback <= 0:
                mean_fallback = 0.1
            a = np.log(mean_fallback)
            b = 0.0
            c = 0.0
            scatter = 0.3
            return np.array([a, b, c, scatter], dtype=float)

        x = centers[valid]
        y = np.log(means_f[valid])
        c, b, a = np.polyfit(x, y, 2)
        resid = y - (a + b * x + c * x * x)
        scatter_fit = np.nanstd(resid)

        stds_pos = np.isfinite(stds_f[valid]) & (stds_f[valid] > 0)
        scatter_from_bin_std = np.nan
        if np.any(stds_pos):
            ratio = stds_f[valid][stds_pos] / means_f[valid][stds_pos]
            sigma2_log = np.log1p(np.maximum(ratio, 1e-6) ** 2)
            scatter_from_bin_std = float(np.nanmedian(np.sqrt(sigma2_log)))

        candidate = np.array([scatter_fit, scatter_from_bin_std], dtype=float)
        candidate = candidate[np.isfinite(candidate) & (candidate > 0)]
        if candidate.size == 0:
            scatter = 0.3
        else:
            scatter = float(np.nanmax(candidate))
        scatter = max(scatter, 1e-3)

        return np.array([a, b, c, scatter], dtype=float)

    def _prepare_sigma_mag_lognormal_params(self):
        """Prepare per-filter [a, b, c, scatter] and interpolation curves (mean + std) for sigma(mag)."""
        n_filters = self.mean_sigma_obs.shape[0]
        params = np.zeros((n_filters, 4), dtype=float)
        ranges = np.zeros((n_filters, 2), dtype=float)
        interp_centers = np.empty(n_filters, dtype=object)
        interp_means = np.empty(n_filters, dtype=object)
        interp_stds = np.empty(n_filters, dtype=object)  # Store stds directly, not scatter
        for filter_idx in range(n_filters):
            centers, _ = self._compute_bin_centers(filter_idx)
            finite_centers = np.asarray(centers, dtype=float)
            finite_centers = finite_centers[np.isfinite(finite_centers)]
            if finite_centers.size == 0:
                ranges[filter_idx] = np.array([20.0, 28.0], dtype=float)
            else:
                ranges[filter_idx] = np.array([
                    float(np.nanmin(finite_centers)),
                    float(np.nanmax(finite_centers)),
                ], dtype=float)

            means_f = np.asarray(self.mean_sigma_obs[filter_idx], dtype=float)
            stds_f = np.asarray(self.stds_sigma_obs[filter_idx], dtype=float)
            valid_interp = (
                np.isfinite(centers)
                & np.isfinite(means_f)
                & np.isfinite(stds_f)
                & (means_f > 0)
            )
            if np.any(valid_interp):
                x = np.asarray(centers[valid_interp], dtype=float)
                y_mean = np.asarray(means_f[valid_interp], dtype=float)
                y_std = np.asarray(stds_f[valid_interp], dtype=float)
                # Cap std/mean ratio to prevent wild extrapolation while
                # preserving broader per-bin scatter seen in real data.
                ratio_capped = np.minimum(y_std / np.maximum(y_mean, 1e-12), 0.8)
                y_std = ratio_capped * y_mean
                order = np.argsort(x)
                x = x[order]
                y_mean = y_mean[order]
                y_std = y_std[order]
                x_unique, idx_unique = np.unique(x, return_index=True)
                y_mean_unique = y_mean[idx_unique]
                y_std_unique = y_std[idx_unique]
                interp_centers[filter_idx] = x_unique
                interp_means[filter_idx] = y_mean_unique
                interp_stds[filter_idx] = np.maximum(y_std_unique, 1e-4)  # Non-zero stds
            else:
                interp_centers[filter_idx] = np.array([], dtype=float)
                interp_means[filter_idx] = np.array([], dtype=float)
                interp_stds[filter_idx] = np.array([], dtype=float)

            params[filter_idx] = self._fit_sigma_mag_lognormal_params(filter_idx)
        self.noise_sigma_mag_params = params
        self.noise_sigma_mag_ranges = ranges
        self.noise_sigma_mag_interp_centers = interp_centers
        self.noise_sigma_mag_interp_means = interp_means
        self.noise_sigma_mag_interp_stds = interp_stds
        # Store the actual training domain bounds (faint & bright limits of observed bins)
        valid_min = np.full(n_filters, np.nan)
        valid_max = np.full(n_filters, np.nan)
        for filter_idx in range(n_filters):
            c = interp_centers[filter_idx]
            if c is not None and len(c) >= 2:
                valid_min[filter_idx] = float(np.nanmin(c))
                valid_max[filter_idx] = float(np.nanmax(c))
        self.noise_sigma_mag_valid_min = valid_min
        self.noise_sigma_mag_valid_max = valid_max

    def _fit_sigma_flux_alpha(self, filter_idx):
        """Estimate source-noise coefficient alpha from empirical sigma-vs-mag trends."""
        centers, means_f = self._compute_bin_centers(filter_idx)
        centers = np.asarray(centers, dtype=float)
        means_f = np.asarray(means_f, dtype=float)

        flux_centers = np.asarray(mag_conversion(centers, convert_to='flux'), dtype=float)
        sigma_flux_emp = (np.log(10) / 2.5) * flux_centers * np.abs(means_f)
        sigma_bkg = float(self.limits[filter_idx])

        sigma_src2 = sigma_flux_emp ** 2 - sigma_bkg ** 2
        valid = (
            np.isfinite(flux_centers) & (flux_centers > 0)
            & np.isfinite(sigma_src2) & (sigma_src2 > 0)
            & (flux_centers > 3.0 * sigma_bkg)
        )

        if not np.any(valid):
            valid = (
                np.isfinite(flux_centers) & (flux_centers > 0)
                & np.isfinite(sigma_src2) & (sigma_src2 > 0)
            )
        if not np.any(valid):
            return 0.0

        alpha_vals = np.sqrt(sigma_src2[valid]) / np.sqrt(flux_centers[valid])
        alpha_vals = alpha_vals[np.isfinite(alpha_vals) & (alpha_vals >= 0)]
        if alpha_vals.size == 0:
            return 0.0
        return float(np.nanmedian(alpha_vals))

    def _prepare_sigma_flux_alpha(self):
        """Prepare per-filter alpha values for sigma_flux^2 = sigma_bkg^2 + alpha^2 flux."""
        n_filters = self.mean_sigma_obs.shape[0]
        alpha = np.zeros(n_filters, dtype=float)
        for filter_idx in range(n_filters):
            alpha[filter_idx] = self._fit_sigma_flux_alpha(filter_idx)
        self.noise_sigma_flux_alpha = alpha

    def _sample_sigma_mag_lognormal(self, mags_det, filter_idx):
        """Sample sigma with safe lognormal parameterization using interpolated mean and std.
        
        Step 1: mean_sigma = interpolate(mag)
        Step 2: std_sigma = interpolate(mag) [with std/mean <= 0.5]
        Step 3: sigma_mag ~ lognormal(mu, sigma) where:
                  sigma2 = log1p((std/mean)^2)
                  mu = log(mean) - 0.5*sigma2
        Step 4: floor to noise_sigma_floor
        """
        if self.noise_sigma_mag_params is None:
            self._prepare_sigma_mag_lognormal_params()

        a, b, c, scatter = self.noise_sigma_mag_params[filter_idx]
        mags_det = np.asarray(mags_det, dtype=float)

        centers = None if self.noise_sigma_mag_interp_centers is None else self.noise_sigma_mag_interp_centers[filter_idx]
        means = None if self.noise_sigma_mag_interp_means is None else self.noise_sigma_mag_interp_means[filter_idx]
        stds = None if self.noise_sigma_mag_interp_stds is None else self.noise_sigma_mag_interp_stds[filter_idx]

        # NOTE: caller is responsible for only passing mags within the valid training domain.
        # No clipping/extrapolation here — out-of-domain mags must be handled upstream.

        # Use interpolation-based parameterization if available
        if centers is not None and means is not None and stds is not None and len(centers) >= 2:
            # Interpolate within the trained range (np.interp clamps at boundary — safe for
            # in-domain mags; out-of-domain mags should not reach here)
            mean_sigma = np.interp(mags_det, centers, means)
            std_sigma = np.interp(mags_det, centers, stds)
            # Ensure std/mean is capped
            std_sigma = np.minimum(std_sigma, 0.5 * mean_sigma)
            std_sigma = np.maximum(std_sigma, 1e-4)

            # Safe lognormal parameterization: E[X]=mean, Var[X]=std^2
            ratio = std_sigma / np.maximum(mean_sigma, 1e-12)
            sigma2 = np.log1p(ratio ** 2)
            mu = np.log(np.maximum(mean_sigma, 1e-12)) - 0.5 * sigma2
            sigma_param = np.sqrt(np.maximum(sigma2, 1e-6))
            mag_errs_det = np.random.lognormal(mean=mu, sigma=sigma_param)
        else:
            # Fallback to quadratic parameterization
            mu = a + b * mags_det + c * mags_det * mags_det
            mag_errs_det = np.random.lognormal(mean=mu, sigma=scatter)

        # Floor
        mag_errs_det = np.maximum(mag_errs_det, self.noise_sigma_floor)
        mag_errs_det = np.clip(mag_errs_det, self.noise_sigma_mag_min, self.noise_sigma_mag_max)

        return mag_errs_det

    def _sample_sigma_empirical(self, filter_idx, bin_indices):
        """Sample sigma directly from stored empirical per-bin values."""
        mag_errs_det = np.full(len(bin_indices), np.nan, dtype=float)

        for bin_idx in np.unique(bin_indices):
            choose_mask = bin_indices == bin_idx
            vals = np.asarray(self.sigma_samples_obs[filter_idx, bin_idx], dtype=float).ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size > 0:
                mag_errs_det[choose_mask] = np.random.choice(vals, size=np.sum(choose_mask), replace=True)

        missing = ~np.isfinite(mag_errs_det)
        if np.any(missing):
            pooled_vals = []
            for bin_idx in range(self.sigma_samples_obs.shape[1]):
                vals = np.asarray(self.sigma_samples_obs[filter_idx, bin_idx], dtype=float).ravel()
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size > 0:
                    pooled_vals.append(vals)
            if len(pooled_vals) > 0:
                pooled_vals = np.concatenate(pooled_vals)
                mag_errs_det[missing] = np.random.choice(pooled_vals, size=np.sum(missing), replace=True)

        return mag_errs_det

    def _draw_detection_mask(self, flux_obs, sigma_flux, limit_flux):
        """Return detection mask using either a hard threshold or a smooth S/N transition."""
        flux_obs = np.asarray(flux_obs, dtype=float)
        sigma_flux = np.asarray(sigma_flux, dtype=float)

        finite = np.isfinite(flux_obs) & np.isfinite(sigma_flux) & (sigma_flux > 0)
        detected = np.zeros_like(flux_obs, dtype=bool)

        if not np.any(finite):
            return detected

        if self.noise_detection_model == 'probabilistic':
            snr_obs = flux_obs[finite] / np.maximum(sigma_flux[finite], 1e-12)
            snr_threshold = 1.0
            snr_width = 1.0
            delta_snr = snr_obs - snr_threshold
            p_detect = 0.5 * (1.0 + np.tanh(delta_snr / snr_width))
            p_detect = np.clip(p_detect, 0.0, 1.0)
            draws = np.random.random(size=p_detect.shape)
            detected[finite] = draws < p_detect
        else:
            detected[finite] = flux_obs[finite] > limit_flux

        return detected

    def simulate(self, mass_max=12, mass_min=4, sfr_prior_type='SFRflat', 
                 sfr_min=-9, sfr_max=2, ssfr_min=-12.0, ssfr_max=-7.5, 
                 z_prior='flat', z_min=0.0, z_max=10.0, Z_min=-2.27, Z_max=0.4, 
                 dust_model='Calzetti', dust_prior='flat', Av_min=0.0, Av_max=3.0, 
                 tx_alpha=1.0, Nparam=3):
        """
        Simulate a galaxy population using specified priors.

        Parameters
        ----------
        mass_max : float, optional
            Maximum log stellar mass (default: 12)
        mass_min : float, optional
            Minimum log stellar mass (default: 4)
        sfr_prior_type : str, optional
            Type of SFR prior: 'SFRflat', 'sSFRflat', 'sSFRlognormal' (default: 'SFRflat')
        sfr_min : float, optional
            Minimum log star formation rate (default: -9)
        sfr_max : float, optional
            Maximum log star formation rate (default: 2)
        ssfr_min : float, optional
            Minimum log specific star formation rate (default: -12.0)
        ssfr_max : float, optional
            Maximum log specific star formation rate (default: -7.5)
        z_prior : str, optional
            Type of redshift prior: 'flat', 'exp' (default: 'flat')
        z_min : float, optional
            Minimum redshift (default: 0.0)
        z_max : float, optional
            Maximum redshift (default: 10.0)
        Z_min : float, optional
            Minimum metallicity [M/H] (default: -2.27)
        Z_max : float, optional
            Maximum metallicity [M/H] (default: 0.4)
        dust_model : str, optional
            Dust attenuation model: 'Calzetti' (default: 'Calzetti')
        dust_prior : str, optional
            Type of dust prior: 'flat' (default: 'flat')
        Av_min : float, optional
            Minimum dust attenuation A_V (default: 0.0)
        Av_max : float, optional
            Maximum dust attenuation A_V (default: 3.0)
        tx_alpha : float, optional
            Alpha parameter for Dirichlet SFH prior (default: 1.0)
        Nparam : int, optional
            Number of SFH parameters for Dirichlet prior (default: 3)

        Notes
        -----
        This method sets up priors using dense_basis and generates a library of
        simulated galaxies. For parametric SFH (τ-delayed), it uses generate_atlas_parametric.
        For non-parametric SFH, it uses dense_basis.generate_atlas with Dirichlet priors.
        """
        import dense_basis as db

        self.refresh_filter_metadata()
        dense_basis_filter_list = self._prepare_dense_basis_filter_list()
        
        # Set up priors
        priors = db.Priors()
        priors.mass_max = mass_max
        priors.mass_min = mass_min
        priors.sfr_prior_type = sfr_prior_type
        priors.sfr_min = sfr_min
        priors.sfr_max = sfr_max
        priors.ssfr_min = ssfr_min
        priors.ssfr_max = ssfr_max
        priors.z_prior = z_prior
        priors.z_min = z_min
        priors.z_max = z_max
        priors.Z_min = Z_min
        priors.Z_max = Z_max
        priors.dust_model = dust_model
        priors.dust_prior = dust_prior
        priors.Av_min = Av_min
        priors.Av_max = Av_max
        priors.tx_alpha = tx_alpha
        priors.Nparam = Nparam

        # Generate atlas based on SFH type
        if self.parametric:
            print("Generating parametric (τ-delayed) SFH atlas...")
            generate_atlas_parametric(
                priors, N_pregrid=self.n_simulation,
                fname=self.atlas_name, store=True, path=self.atlas_path,
                filter_list=dense_basis_filter_list, filt_dir=self.filter_path, 
                norm_method='none'
            )
        else:
            print("Generating non-parametric (Dirichlet) SFH atlas...")
            db.generate_atlas(
                N_pregrid=self.n_simulation, priors=priors,
                fname=self.atlas_name, store=True, path=self.atlas_path,
                filter_list=dense_basis_filter_list, filt_dir=self.filter_path, 
                norm_method='none'
            )

    def load_simulation(self):
        """
        Load the simulated galaxy population from saved atlas.

        Returns
        -------
        obs : np.ndarray
            Observed magnitudes (n_simulation, n_filters)
        theta : np.ndarray
            Physical parameters (n_simulation, n_params)

        Notes
        -----
        Updates self.obs, self.theta, and self.labels based on the SFH type.
        For parametric SFH, parameters are [M*, M*_formed, SFR, τ, t_i, [M/H], A_V, z].
        For Dirichlet SFH, parameters are [M*, SFR, t_25%, t_50%, t_75%, [M/H], A_V, z].
        """
        import dense_basis as db
        
        # Determine number of SFH parameters
        nparam = 2 if self.parametric else 3
        
        # Load atlas
        atlas = db.load_atlas(
            self.atlas_name, N_pregrid=self.n_simulation, 
            N_param=nparam, path=self.atlas_path
        )
        
        # Extract SEDs and convert to magnitudes
        atlas_seds = atlas['sed']

        def _as_1d(arr):
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return np.array([arr.item()])
            if arr.ndim == 1:
                return arr
            return arr[:, 0]

        def _to_log10_if_linear(arr):
            arr = _as_1d(arr).astype(float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return arr
            # If values include non-positive entries, treat as already log10.
            # Physical linear M* and SFR must be > 0.
            if np.nanmin(finite) <= 0:
                return arr
            return np.log10(np.clip(arr, 1e-300, None))

        def _to_log10_mstar(arr):
            arr = _as_1d(arr).astype(float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return arr
            if np.nanmin(finite) <= 0:
                return arr
            # Stellar masses already in log10(Msun) are typically O(1e1),
            # while linear masses are O(1e6-1e12) Msun.
            if np.nanpercentile(finite, 99) < 100.0:
                return arr
            return np.log10(np.clip(arr, 1e-300, None))
        
        # Apply filter removal if specified
        if self.remove_filters is not None:
            atlas_seds = atlas_seds[:, [i for i in range(len(atlas_seds[0,:]))
                                       if i not in self.remove_filters]]
        
        # Convert from microJy to AB magnitudes
        obs = -2.5 * np.log10(atlas_seds * 1e-6 / 3631)
        n_loaded = obs.shape[0]

        # Extract parameters based on SFH type
        if self.parametric:
            sfhs = atlas['sfh_tuple']
            theta = np.zeros((n_loaded, 8))
            # Build key physical parameters explicitly from atlas keys.
            # Convert to log10 only when arrays are in linear units.
            theta[:, 0] = _to_log10_mstar(atlas['mstar'])  # log M* (surviving)
            theta[:, 2] = _to_log10_if_linear(atlas['sfr'])    # log SFR
            # Keep formed-mass, tau and ti from sfh tuple in dense_basis atlas.
            theta[:, 1] = sfhs[:, 1]  # log M* (formed)
            theta[:, 3] = np.clip(sfhs[:, 3], 0.1, 5.0)   # τ [Gyr], clipped to [0.1,5]
            theta[:, 4] = sfhs[:, 4]  # t_i
            theta[:, 5] = _as_1d(atlas['met'])  # [M/H]
            theta[:, 6] = _as_1d(atlas['dust'])  # A_V
            theta[:, 7] = _as_1d(atlas['zval'])  # z
            
            self.labels = [
                'log($\\rm{M}_{*}/\\rm{M}_{\\odot}$)',
                'log($\\rm{M}_{*}^{\\rm{formed}}/\\rm{M}_{\\odot}$)',
                'log(SFR/($\\rm{M}_{\\odot}$/yr))',
                '$\\tau$ [Gyr]', '$t_i$ [Gyr]',
                '[M/H]', 'Av', 'z'
            ]
        else:
            # Dirichlet SFH parameters
            theta = np.zeros((n_loaded, 8))
            sfhs = atlas['sfh_tuple_rec']
            sfhs = np.reshape(sfhs, (n_loaded, 6))
            theta[:, 0] = _to_log10_mstar(atlas['mstar'])  # log M* (surviving)
            theta[:, 1] = _to_log10_if_linear(atlas['sfr'])    # log SFR
            theta[:, 2] = sfhs[:, 3]  # t_25%
            theta[:, 3] = sfhs[:, 4]  # t_50%
            theta[:, 4] = sfhs[:, 5]  # t_75%
            theta[:, 5] = _as_1d(atlas['met'])  # [M/H]
            theta[:, 6] = _as_1d(atlas['dust'])  # A_V
            theta[:, 7] = _as_1d(atlas['zval'])  # z

            # Add formed stellar mass if requested
            if self.both_masses:
                theta = np.concatenate((_to_log10_mstar(atlas['mstar']).reshape(-1,1), theta), axis=1)

        self.obs = obs
        self.theta = theta

        # Apply a physical validity mask to remove clearly unphysical atlas entries
        logM = _to_log10_mstar(atlas['mstar'])
        logSFR = _to_log10_if_linear(atlas['sfr'])
        z = _as_1d(atlas['zval'])

        # Magnitude sanity: only require a finite median magnitude.
        # A previous hard cut (15 < median_mag < 35) removed a large population
        # of faint, high-redshift galaxies and substantially weakened the
        # mass–flux relation in the retained atlas, which in turn biased
        # inference high at z >~ 1. Keep all finite SEDs here and let the
        # downstream noise/detection model define the observable sample.
        median_mag_per_gal = np.nanmedian(obs, axis=1)  # obs already in AB mag
        mag_sane = np.isfinite(median_mag_per_gal)

        # t_i < age of the universe at z (cosmological causality)
        try:
            from astropy.cosmology import FlatLambdaCDM as _FlatLCDM
            _cosmo = _FlatLCDM(H0=70, Om0=0.3)
            z_safe = np.clip(z, 0.01, 20.0)
            age_at_z = np.asarray(_cosmo.age(z_safe).value, dtype=float)  # Gyr
            if self.parametric:
                t_i_col = self.theta[:, 4]
            else:
                t_i_col = np.zeros(len(self.theta))   # Dirichlet: no t_i column
            ti_ok = np.isfinite(t_i_col) & (t_i_col < age_at_z) & (t_i_col > 0)
        except Exception:
            ti_ok = np.ones(n_loaded, dtype=bool)   # skip if astropy unavailable

        logm_ok = (np.isfinite(logM) & (logM > 5) & (logM < 13)).ravel()
        logsfr_ok = (np.isfinite(logSFR) & (logSFR > -15) & (logSFR < 5)).ravel()
        z_ok = np.isfinite(z).ravel()
        mag_sane = np.asarray(mag_sane, dtype=bool).ravel()
        ti_ok = np.asarray(ti_ok, dtype=bool).ravel()

        if not (
            logm_ok.size == logsfr_ok.size == z_ok.size == mag_sane.size == ti_ok.size == n_loaded
        ):
            raise ValueError(
                "Mask shape mismatch in physical validity filters: "
                f"logm_ok={logm_ok.shape}, logsfr_ok={logsfr_ok.shape}, "
                f"z_ok={z_ok.shape}, mag_sane={mag_sane.shape}, "
                f"ti_ok={ti_ok.shape}, expected N={n_loaded}"
            )

        combined = logm_ok & logsfr_ok
        if DEBUG_MASK_DIAGNOSTICS:
            # Step-by-step diagnostics to identify over-constrained mask intersections
            print(f"N total: {n_loaded}")
            print(f"logm_ok: {np.sum(logm_ok)}")
            print(f"logsfr_ok: {np.sum(logsfr_ok)}")
            print(f"z_ok: {np.sum(z_ok)}")
            print(f"mag_sane: {np.sum(mag_sane)}")
            print(f"ti_ok: {np.sum(ti_ok)}")
            print(f"logm_ok & logsfr_ok: {np.sum(combined)}")
        combined = combined & z_ok
        if DEBUG_MASK_DIAGNOSTICS:
            print(f"(logm_ok & logsfr_ok) & z_ok: {np.sum(combined)}")
        combined = combined & mag_sane
        if DEBUG_MASK_DIAGNOSTICS:
            print(f"... & mag_sane: {np.sum(combined)}")
        combined = combined & ti_ok
        if DEBUG_MASK_DIAGNOSTICS:
            print(f"final combined: {np.sum(combined)}")

        valid = combined

        self.theta = self.theta[valid]
        self.obs = self.obs[valid]
        self.n_simulation = int(np.sum(valid))
        print(
            f"Physical mask applied: remaining {len(self.theta)} / {len(valid)}  "
            f"(mag_sane={mag_sane.sum()}, ti_ok={ti_ok.sum()}, combined={valid.sum()})"
        )

        return self.obs, self.theta
    
    def load_obs_features(self):
        """
        Loads observational features from pre-saved numpy files.

        Parameters:
        None

        Returns:
        None
        """
        # Load observational features from the survey

        #mean of the distribution of noise in the galaxies for each filter and different bins of flux
        self.mean_sigma_obs = np.load(os.path.join(self.filter_path, self.mean_sigma_file)) 
        #std of the distribution of noise in the galaxies for each filter and different bins of flux
        self.stds_sigma_obs = np.load(os.path.join(self.filter_path, self.std_sigma_file)) 
        sigma_samples_file = self.sigma_samples_file
        if sigma_samples_file is None and self.std_sigma_file.startswith('std_sigma_'):
            sigma_samples_file = self.std_sigma_file.replace('std_sigma_', 'sigma_samples_', 1)
        sigma_samples_path = None if sigma_samples_file is None else os.path.join(self.filter_path, sigma_samples_file)
        if sigma_samples_path is not None and os.path.exists(sigma_samples_path):
            self.sigma_samples_obs = np.load(sigma_samples_path, allow_pickle=True)
        else:
            self.sigma_samples_obs = None
        print('Sigma samples loaded:', self.sigma_samples_obs is not None)
        #different bins of flux for each filter
        self.percentiles = np.load(os.path.join(self.filter_path, self.percentiles_file))
        #1 sigma depth limits for each filter
        self.limits=np.load(os.path.join(self.filter_path, self.limits_file)) 

        self.refresh_filter_metadata()
        n_filters = self.n_filters
        self.mean_sigma_obs = self.mean_sigma_obs[:n_filters, :]
        self.stds_sigma_obs = self.stds_sigma_obs[:n_filters, :]
        if self.sigma_samples_obs is not None:
            self.sigma_samples_obs = self.sigma_samples_obs[:n_filters, :]
        self.percentiles = self.percentiles[:, :n_filters]
        self.limits = self.limits[:n_filters]
        
        if self.remove_filters is not None:
            keep = [i for i in range(self.mean_sigma_obs.shape[0]) if i not in self.remove_filters]
            self.mean_sigma_obs = self.mean_sigma_obs[keep, :]
            self.stds_sigma_obs = self.stds_sigma_obs[keep, :]
            if self.sigma_samples_obs is not None:
                self.sigma_samples_obs = self.sigma_samples_obs[keep, :]
            self.percentiles = self.percentiles[:, keep]
            self.limits = self.limits[keep]

        self._prepare_sigma_mag_lognormal_params()
        self._prepare_sigma_flux_alpha()

        print('Observational features loaded')        


    def add_noise_nan_limit_all_old(self):
        """
        Add realistic observational effects to all simulated galaxies.
        
        This includes:
        - Photometric uncertainties based on magnitude
        - Detection limits
        - Non-detections
        
        Updates self.mag with processed photometry ready for training.
        """
        self.mag = np.zeros((self.n_simulation, len(self.obs[0,:]), 2))

        for j in trange(self.n_simulation, desc="Adding observational realism"):
            for i in range(len(self.obs[0,:])):
                self.mag[j, i, :] = self._add_noise_nan_limit(self.obs[j][i], i)

    def _add_noise_nan_limit_old(self, mag, filter_idx):
        """
        Add noise and handle detection limits for a single magnitude measurement.

        Parameters
        ----------
        mag : float
            Input magnitude
        filter_idx : int
            Index of the filter

        Returns
        -------
        list
            [noisy_magnitude, uncertainty]
        """
        # Magnitude-dependent uncertainty bins
        i_not_last_bin = [10,12,13,14,15,16,17,18]
        percentiles = self.percentiles
        flux = mag_conversion(mag, convert_to='flux')

        # Determine uncertainty based on magnitude and apply detection limit
        if self.include_limit and flux > self.limits[filter_idx]:
            # Assign uncertainty based on magnitude bin
            if mag < percentiles[0, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 0], 
                    self.stds_sigma_obs[filter_idx, 0]
                )
            elif mag < percentiles[1, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 1], 
                    self.stds_sigma_obs[filter_idx, 1]
                )
            elif mag < percentiles[2, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 2], 
                    self.stds_sigma_obs[filter_idx, 2]
                )
            else:
                bin_idx = 2 if filter_idx in i_not_last_bin else 3
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, bin_idx], 
                    self.stds_sigma_obs[filter_idx, bin_idx]
                )

            # Add noise
            noise = np.random.normal(0.0, np.abs(mag_err))
            mag_n_noise = mag + noise
        else:
            # Non-detection
            mag_n_noise = 0.0
            mag_err = mag_conversion(self.limits[filter_idx], convert_to='mag')

        if self.include_sigma:
            return [mag_n_noise, mag_err]
        else:
            return [mag, mag_err]
        
    
    def add_noise_nan_limit_all(self):
        """
        Add realistic observational effects to all simulated galaxies (Vectorized).
        
        This includes:
        - Photometric uncertainties based on magnitude
        - Detection limits
        - Non-detections (set to 99.0)
        
        Updates self.mag with processed photometry ready for training.
        """
        n_sims, n_filts = self.obs.shape
        self.mag = np.zeros((n_sims, n_filts, 2))
        if DEBUG_NOISE_ONLY and not DEBUG_SIGMA_ONLY:
            self._debug_noise_pulls = []

        if DEBUG_SIGMA_ONLY:
            print('DEBUG_SIGMA_ONLY=True: flux noise and detection cuts are bypassed')
        elif DEBUG_NOISE_ONLY:
            print('DEBUG_NOISE_ONLY=True: flux noise ON, detection cuts bypassed')
        
        # We loop over FILTERS instead of GALAXIES. 
        # This allows us to process all simulations instantly using numpy arrays.
        for i in trange(n_filts, desc=f"Adding observational realism to {n_filts} filters"):
            
            # Extract the entire column of magnitudes for this filter
            mags_filter = self.obs[:, i]
            
            # Process all galaxies for this filter at once
            mag_n_noise, mag_err = self._add_noise_nan_limit(mags_filter, i)
            
            # Store results
            self.mag[:, i, 0] = mag_n_noise
            self.mag[:, i, 1] = mag_err

        if DEBUG_NOISE_ONLY and not DEBUG_SIGMA_ONLY:
            pull_chunks = getattr(self, "_debug_noise_pulls", None)
            if pull_chunks:
                pull_all = np.concatenate(pull_chunks)
                pull_all = pull_all[np.isfinite(pull_all)]
                if pull_all.size > 0:
                    p16, p50, p84 = np.percentile(pull_all, [16, 50, 84])
                    frac1 = np.mean(np.abs(pull_all) <= 1.0)
                    frac2 = np.mean(np.abs(pull_all) <= 2.0)
                    print(
                        "[noise-only global] "
                        f"pull_mean={np.mean(pull_all):.3f} pull_std={np.std(pull_all):.3f} "
                        f"p16={p16:.3f} p50={p50:.3f} p84={p84:.3f} "
                        f"|pull|<=1: {100*frac1:.1f}% |pull|<=2: {100*frac2:.1f}% "
                        f"N={pull_all.size}"
                    )

    def _add_noise_nan_limit_mag_space(self, mag_array, filter_idx):
        """
        LEGACY: noise added in magnitude space (kept for reference).
        Vectorized noise addition and detection limits for an array of magnitudes.

        Parameters
        ----------
        mag_array : np.ndarray
            Array of input magnitudes for a single filter (all galaxies).
        filter_idx : int
            Index of the filter.

        Returns
        -------
        tuple
            (noisy_magnitudes_array, uncertainties_array)
        """
        
        
        flux_array = mag_conversion(mag_array, convert_to='flux')

        # 1. Determine limits
        limit_flux = self.limits[filter_idx] #in microjy
        limit_mag_err = mag_conversion(limit_flux, convert_to='mag')

        # 2. Setup defaults for non-detections (Using 99.0)
        final_mags = np.full_like(mag_array, 99.0)
        final_errs = np.full_like(mag_array, limit_mag_err)

        valid_mask = np.isfinite(flux_array) & (flux_array > 0)

        mags_valid = mag_array[valid_mask]

        if mags_valid.size > 0:
            # 4. Find magnitude bins dynamically (replaces the slow if/elif chain)
            bin_indices = np.zeros(len(mags_valid), dtype=int)
            use_empirical_sigma = (
                self.noise_sigma_sampler == 'empirical' and self.sigma_samples_obs is not None
            )
            use_mag_lognormal = self.noise_sigma_sampler == 'mag_lognormal'

            if use_mag_lognormal:
                mag_errs_det = np.asarray(
                    self._sample_sigma_mag_lognormal(mags_valid, filter_idx),
                    dtype=float,
                )
            else:
                percentiles_f = self.percentiles[:, filter_idx]
                bin_indices = np.digitize(mags_valid, percentiles_f)

                # Clip to max bin index to avoid IndexError for very faint/bright sources
                max_bin = self.mean_sigma_obs.shape[1] - 1
                bin_indices = np.clip(bin_indices, 0, max_bin)

                # 5. Extract statistics for the corresponding bins
                pixel_means = self.mean_sigma_obs[filter_idx, bin_indices]
                pixel_stds = self.stds_sigma_obs[filter_idx, bin_indices]

                # 6. Sample σ values from the configured distribution
                if use_empirical_sigma:
                    mag_errs_det = np.asarray(
                        self._sample_sigma_empirical(filter_idx, bin_indices),
                        dtype=float,
                    )
                else:
                    mag_errs_det = np.asarray(self._sample_sigma_distribution(pixel_means, pixel_stds), dtype=float)

            finite_sigma = np.isfinite(mag_errs_det) & (mag_errs_det > 0)
            if not np.any(finite_sigma):
                return final_mags, final_errs

            mags_valid = mags_valid[finite_sigma]
            mag_errs_det = mag_errs_det[finite_sigma]

            # 7. Add noise
            noise = np.random.normal(0.0, mag_errs_det)
            mags_measured = mags_valid + noise
            flux_measured = mag_conversion(mags_measured, convert_to='flux')
            sigma_flux = np.abs(np.log(10) / 2.5 * np.maximum(flux_measured, 1e-12) * mag_errs_det)

            if self.include_limit:
                detected_after_noise = self._draw_detection_mask(flux_measured, sigma_flux, limit_flux)
            else:
                detected_after_noise = np.ones_like(mags_valid, dtype=bool)

            target_idx = np.where(valid_mask)[0][finite_sigma]
            det_idx = target_idx[detected_after_noise]

            if self.include_sigma:
                final_mags[det_idx] = mags_measured[detected_after_noise]
            else:
                final_mags[det_idx] = mags_valid[detected_after_noise]

            final_errs[det_idx] = mag_errs_det[detected_after_noise]

        return final_mags, final_errs

    def _add_noise_nan_limit(self, mag_array, filter_idx):
        """
        CORRECT noise model:
        - noise in flux space
        - SNR-based detection
        """

        # --- convert to flux ---
        flux_true = mag_conversion(mag_array, convert_to='flux')

        # --- detection limit (1σ) ---
        sigma_lim = self.limits[filter_idx]

        # --- sample sigma in FLUX space ---
        # mean_sigma_obs shape: (n_filters, n_bins); clip to n_bins - 1
        n_bins = self.mean_sigma_obs.shape[1]
        mags = mag_array

        bin_indices = np.digitize(mags, self.percentiles[:, filter_idx]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # sample σ_mag from configured distribution
        use_empirical_sigma = (
            self.noise_sigma_sampler == 'empirical' and self.sigma_samples_obs is not None
        )
        use_mag_lognormal = self.noise_sigma_sampler == 'mag_lognormal'

        if use_mag_lognormal:
            # ----------------------------------------------------------------
            # Domain-split: only apply the trained σ(mag) model inside the
            # bin range seen in real detected data.  For galaxies fainter than
            # the training edge (or brighter, but that never happens) the
            # noise is purely background-limited: σ_mag → σ_lim.
            # ----------------------------------------------------------------
            if self.noise_sigma_mag_valid_min is None:
                self._prepare_sigma_mag_lognormal_params()

            valid_min = float(self.noise_sigma_mag_valid_min[filter_idx])
            valid_max = float(self.noise_sigma_mag_valid_max[filter_idx])
            if not np.isfinite(valid_min):
                valid_min = -np.inf
            if not np.isfinite(valid_max):
                valid_max = np.inf

            in_domain = (mags >= valid_min) & (mags <= valid_max)

            sigma_mag = np.full_like(mags, np.nan, dtype=float)

            # In-domain: use fitted σ(mag) model
            if np.any(in_domain):
                sigma_mag[in_domain] = self._sample_sigma_mag_lognormal(
                    mags[in_domain], filter_idx
                )

            # Out-of-domain (faint/below-limit): background-noise-limited
            # σ_flux = σ_lim  →  σ_mag = (2.5/ln10) * σ_lim / |flux|
            if np.any(~in_domain):
                flux_faint = np.maximum(np.abs(flux_true[~in_domain]), 1e-12)
                sigma_mag_faint = (2.5 / np.log(10)) * sigma_lim / flux_faint
                # Cap at a sensible maximum to avoid infinite mag errors
                sigma_mag_faint = np.clip(sigma_mag_faint, self.noise_sigma_floor, self.noise_sigma_mag_max)
                sigma_mag[~in_domain] = sigma_mag_faint

            # Debug: print σ_mag percentiles once per call to spot rogue values
            p50, p90, p99, p999 = np.nanpercentile(sigma_mag, [50, 90, 99, 99.9])
            if p999 > 1.0:
                print(
                    f"[σ-debug filter={filter_idx}] p50={p50:.4f} p90={p90:.4f} "
                    f"p99={p99:.4f} p99.9={p999:.4f}  "
                    f"in_domain={in_domain.sum()}/{len(in_domain)}"
                )

            sigma_mag = np.asarray(sigma_mag, dtype=float)
        elif use_empirical_sigma:
            sigma_mag = np.zeros_like(mags)
            for i in range(len(mags)):
                samples = self.sigma_samples_obs[filter_idx, bin_indices[i]]
                vals = np.asarray(samples, dtype=float).ravel()
                if len(vals) > 0:
                    sigma_mag[i] = np.random.choice(vals)
                else:
                    sigma_mag[i] = self.mean_sigma_obs[filter_idx, bin_indices[i]]
        else:
            mean = self.mean_sigma_obs[filter_idx, bin_indices]
            std  = self.stds_sigma_obs[filter_idx, bin_indices]
            sigma_mag = np.random.normal(mean, std)

        # Convert sigma_mag -> sigma_flux and use THIS as the only sigma model.
        sigma_mag = np.asarray(sigma_mag, dtype=float)
        bad_sigma = ~np.isfinite(sigma_mag) | (sigma_mag <= 0)
        if np.any(bad_sigma):
            sigma_mag[bad_sigma] = self.mean_sigma_obs[filter_idx, bin_indices[bad_sigma]]
        sigma_mag = np.maximum(sigma_mag, self.noise_sigma_floor)
        sigma_mag = np.clip(sigma_mag, self.noise_sigma_mag_min, self.noise_sigma_mag_max)
        sigma_flux = (np.log(10) / 2.5) * np.maximum(np.abs(flux_true), 1e-12) * sigma_mag
        # Floor at the background 1σ limit (σ_lim) so that faint galaxies
        # (flux ≪ σ_lim) are noise-dominated and can fall below the SNR
        # threshold → realistic detection curve that drops to ~0 at faint end.
        sigma_flux = np.maximum(sigma_flux, np.maximum(sigma_lim, 1e-12))

        # --- ADD NOISE IN FLUX SPACE ---
        if DEBUG_SIGMA_ONLY:
            flux_obs_raw = flux_true
        else:
            flux_obs_raw = flux_true + np.random.normal(0, sigma_flux)

        # --- detection using SNR on raw (possibly negative) flux ---
        # Apply detection BEFORE any mag conversion; negative flux = non-detection.
        snr = flux_obs_raw / np.maximum(sigma_flux, 1e-12)
        if DEBUG_SIGMA_ONLY or DEBUG_NOISE_ONLY:
            detected = np.ones_like(flux_true, dtype=bool)
        else:
            snr_threshold = getattr(self, 'snr_threshold', 1.0)
            detected = snr >= snr_threshold

        if DEBUG_NOISE_ONLY and not DEBUG_SIGMA_ONLY:
            pull = (flux_obs_raw - flux_true) / np.maximum(sigma_flux, 1e-12)
            pull_ok = np.isfinite(pull)
            if np.any(pull_ok):
                pull_vals = pull[pull_ok]
                if hasattr(self, "_debug_noise_pulls"):
                    self._debug_noise_pulls.append(pull_vals)
                p16, p50, p84 = np.percentile(pull_vals, [16, 50, 84])
                frac1 = np.mean(np.abs(pull_vals) <= 1.0)
                frac2 = np.mean(np.abs(pull_vals) <= 2.0)
                print(
                    f"[noise-only filter={filter_idx}] pull_mean={np.mean(pull_vals):.3f} "
                    f"pull_std={np.std(pull_vals):.3f} p16={p16:.3f} p50={p50:.3f} p84={p84:.3f} "
                    f"|pull|<=1: {100*frac1:.1f}% |pull|<=2: {100*frac2:.1f}%"
                )

        # --- convert to mag ---
        # Production path keeps the physical positive-flux requirement.
        # Debug path bypasses positivity gating to isolate censoring effects.
        if DEBUG_SIGMA_ONLY or DEBUG_NOISE_ONLY:
            det_pos = detected
            flux_for_mag = np.maximum(flux_obs_raw, 1e-12)
        else:
            det_pos = detected & (flux_obs_raw > 0)
            flux_for_mag = flux_obs_raw

        mag_obs = np.full_like(mag_array, np.nan)
        mag_obs[det_pos] = mag_conversion(flux_for_mag[det_pos], convert_to='mag')

        # Debug: SNR distribution for diagnosed bands
        snr_med = float(np.nanmedian(snr))
        snr_p99 = float(np.nanpercentile(snr, 99))
        det_frac = float(det_pos.sum()) / max(len(flux_obs_raw), 1)
        if det_frac < 0.3 or det_frac > 0.99:
            print(
                f"[det-debug filter={filter_idx}] SNR_med={snr_med:.3f} SNR_p99={snr_p99:.2f} "
                f"det_pos={det_pos.sum()}/{len(flux_obs_raw)} ({100*det_frac:.1f}%)"
            )

        # --- outputs ---
        if getattr(self, 'noise_observation_space', 'mag') == 'flux':
            # Flux-space production path: keep the full noisy flux distribution,
            # including negative realizations, and return (flux, sigma_flux).
            final_flux = np.asarray(flux_obs_raw, dtype=float)
            final_sigma_flux = np.asarray(sigma_flux, dtype=float)

            bad_flux = ~np.isfinite(final_flux)
            if np.any(bad_flux):
                final_flux[bad_flux] = 0.0

            bad_sigma_flux = ~np.isfinite(final_sigma_flux) | (final_sigma_flux <= 0)
            if np.any(bad_sigma_flux):
                final_sigma_flux[bad_sigma_flux] = np.maximum(sigma_lim, 1e-12)

            return final_flux, final_sigma_flux

        final_mag = np.full_like(mag_array, 99.0)
        final_err = np.full_like(mag_array, mag_conversion(sigma_lim, convert_to='mag'))

        valid_det = det_pos & np.isfinite(mag_obs)
        final_mag[valid_det] = mag_obs[valid_det]

        # convert σ_flux → σ_mag for storage (use actual observed flux for conversion)
        final_err[valid_det] = (2.5 / np.log(10)) * sigma_flux[valid_det] / np.maximum(np.abs(flux_obs_raw[valid_det]), 1e-12)

        return final_mag, final_err

    def _transform_flux_observation_matrix(self, obs):
        """Map raw flux features to a numerically stable conditioning space."""
        obs = np.asarray(obs, dtype=float)
        if getattr(self, 'noise_observation_space', 'mag') != 'flux':
            return obs

        if self.limits is None:
            raise ValueError("Flux-space observation transform requires self.limits to be loaded.")

        softening = np.maximum(np.asarray(self.limits, dtype=float), 1e-12)

        if obs.ndim != 2:
            raise ValueError(f"Expected 2D observation matrix, got shape {obs.shape}")

        if self.condition_sigma:
            n_expected = 2 * len(softening)
            if obs.shape[1] != n_expected:
                raise ValueError(
                    f"Flux-space observation matrix width mismatch: got {obs.shape[1]}, expected {n_expected}"
                )

            flux = obs[:, 0::2]
            sigma_flux = obs[:, 1::2]
            flux_tx = np.arcsinh(flux / softening[None, :])
            sigma_tx = np.log10(np.maximum(sigma_flux / softening[None, :], 1e-12))

            transformed = np.empty_like(obs, dtype=float)
            transformed[:, 0::2] = flux_tx
            transformed[:, 1::2] = sigma_tx
            return transformed

        if obs.shape[1] != len(softening):
            raise ValueError(
                f"Flux-space observation matrix width mismatch: got {obs.shape[1]}, expected {len(softening)}"
            )

        return np.arcsinh(obs / softening[None, :])

    def _get_conditioning_observations(self):
        """Return training/test conditioning inputs in the configured feature space."""
        if self.condition_sigma:
            obs = np.reshape(self.mag, (self.n_simulation, 2 * len(self.obs[0])))
        elif self.include_mask or self.include_limit or self.include_sigma:
            obs = self.mag[:, :, 0]
        else:
            print('No noise, mask or limit included')
            obs = self.obs if self.mag is None else self.mag[:, :, 0]

        return self._transform_flux_observation_matrix(obs)


    def train(self, min_thetas=[6, -10, 0, 0, 0, -2.3, 0, 0], 
              max_thetas=[12, 3, 1, 1, 1, 0.4, 3, 10], 
              n_max=1000000, epochs_max=None, nblocks=15, nhidden=500, 
              val_fraction=0.1, device='cpu'):
        """
        Train the neural density estimator using simulation-based inference.

        Parameters
        ----------
        min_thetas : list, optional
            Lower bounds for posterior parameters
        max_thetas : list, optional  
            Upper bounds for posterior parameters
        n_max : int, optional
            Maximum number of training samples (default: 1000000)
        epochs_max : int, optional
            Maximum training epochs (default: None, uses early stopping)
        nblocks : int, optional
            Number of coupling blocks in normalizing flow (default: 15)
        nhidden : int, optional
            Number of hidden features per block (default: 500)
        val_fraction : float, optional
            Fraction of data for validation (default: 0.1)
        device : str, optional
            Device for training: 'cpu' or 'cuda' (default: 'cpu')

        Notes
        -----
        Trains a Masked Autoregressive Flow (MAF) using Neural Posterior Estimation.
        Saves the trained model to self.model_path + self.model_name.
        """
        # Prepare observations based on configuration
        obs = self._get_conditioning_observations()

        # Initialize neural network
        maf_model = sbi.neural_nets.posterior_nn(
            'maf', hidden_features=nhidden, num_transforms=nblocks, num_layers=2
        )

        if self.infer_z:
            # Define parameter bounds
            lower_bounds = torch.tensor(min_thetas, dtype=torch.float32)
            upper_bounds = torch.tensor(max_thetas, dtype=torch.float32)
            bounds = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)
            
            print('Lower bounds:', lower_bounds)
            print('Upper bounds:', upper_bounds)

            # Initialize NPE
            anpe = Inference.SNPE(prior=bounds, density_estimator=maf_model, device=device)

            # Add training data
            anpe.append_simulations(
                torch.as_tensor(self.theta[:n_max, :].astype(np.float32)).to(device),
                torch.as_tensor(obs[:n_max, :].astype(np.float32)).to(device)
            )
        else:
            # Training without redshift inference
            lower_bounds = torch.tensor(min_thetas[:-1], dtype=torch.float32)
            upper_bounds = torch.tensor(max_thetas[:-1], dtype=torch.float32)
            bounds = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)
            
            anpe = Inference.SNPE(prior=bounds, density_estimator=maf_model, device=device)
            
            # Concatenate redshift as input
            obs_with_z = np.concatenate([
                obs[:n_max, :], 
                np.reshape(self.theta[:n_max, -1], (n_max, 1))
            ], axis=1)
            
            anpe.append_simulations(
                torch.as_tensor(self.theta[:n_max, :-1].astype(np.float32)).to(device),
                torch.as_tensor(obs_with_z.astype(np.float32)).to(device)
            )

        # Train
        train_kwargs = {
            'show_train_summary': True, 
            'retrain_from_scratch': True,
            'validation_fraction': val_fraction
        }
        if epochs_max is not None:
            train_kwargs['max_num_epochs'] = epochs_max

        p_theta_x_est = anpe.train(**train_kwargs)

        # Build posterior and save
        qphi = anpe.build_posterior(p_theta_x_est)
        
        model_file = self.model_path + self.model_name
        anpe_file = self.model_path + 'anpe_' + self.model_name

        with open(model_file, "wb") as f:
            pickle.dump(qphi, f)
        with open(anpe_file, "wb") as f:
            pickle.dump(anpe, f)

        print(f"Model saved to {model_file}")

    def test_performance(self, n_test=1000, n_samples=100, return_posterior=False, device='cpu', sample_with='rejection'):
        """
        Test model performance on held-out simulations.

        Parameters
        ----------
        n_test : int, optional
            Number of test samples (default: 1000)
        n_samples : int, optional
            Number of posterior samples per test case (default: 100)
        return_posterior : bool, optional
            Whether to return full posterior samples (default: False)
        device : str, optional
            Device for inference (default: 'cpu')
        sample_with : str, optional
            Posterior sampling method passed to sbi posterior sampler
            (default: 'rejection'; alternative: 'mcmc')

        Returns
        -------
        posteriors : np.ndarray, optional
            Full posterior samples if return_posterior=True

        Notes
        -----
        Updates self.means_test, self.stds_test with test results.
        """
        # Load trained model
        with open(self.model_path + self.model_name, 'rb') as f:
            qphi = pickle.load(f)

        # sbi>=0.18 requires setting sampling backend in build_posterior(), not in sample()
        if sample_with != 'rejection':
            anpe_file = self.model_path + 'anpe_' + self.model_name
            try:
                with open(anpe_file, 'rb') as f:
                    anpe = pickle.load(f)
                qphi = anpe.build_posterior(sample_with=sample_with)
                print(f"Using posterior sampler backend: {sample_with}")
            except Exception as exc:
                print(
                    f"WARNING: could not rebuild posterior with sample_with='{sample_with}' "
                    f"from {anpe_file} ({exc}). Falling back to default sampler."
                )

        means, stds, posteriors, modes = [], [], [], []

        # Prepare test observations
        obs = self._get_conditioning_observations()

        if not self.infer_z:
            obs = np.concatenate([obs, np.reshape(self.theta[:, -1], (len(obs), 1))], axis=1)

        # Run inference on test set
        for j in trange(n_test, desc="Testing performance"):
            posterior_samples = np.array(
                qphi.sample(
                    (n_samples,), 
                    x=torch.as_tensor(obs[j].astype(np.float32)).to(device),
                    show_progress_bars=False
                ).detach().to('cpu')
            )
            
            posteriors.append(posterior_samples)
            stds.append(np.std(posterior_samples, axis=0))
            means.append(np.median(posterior_samples, axis=0))
            modes.append(stats.mode(np.round(posterior_samples, 1), axis=0))

        self.means_test = np.array(means)
        self.stds_test = np.array(stds)
        self.mode_test = np.array(modes)

        if return_posterior:
            return np.array(posteriors)
        

    def _get_posterior_obs(self, obs, qphi, n_samples=1000, bar=True, input_z=None,device='cpu'):
        """
        Generate posterior samples for observed data.

        Parameters:
        - obs: numpy array
            Observed data.
        - qphi: object
            Trained model for sampling.
        - n_samples: int, optional (default=1000)
            Number of samples to generate.
        - bar: bool, optional (default=True)
            Whether to show a progress bar.
        - input_z: numpy array, optional
            Input redshift values.

        Returns:
        - numpy array
            Posterior samples.
        """
        obs = self._transform_flux_observation_matrix(obs)

        if not self.infer_z and input_z is not None:
            input_z = np.asarray(input_z)
            if input_z.ndim == 0:
                z_col = np.full((len(obs), 1), float(input_z), dtype=float)
            else:
                input_z = np.ravel(input_z)
                if len(input_z) == 1:
                    z_col = np.full((len(obs), 1), float(input_z[0]), dtype=float)
                elif len(input_z) == len(obs):
                    z_col = np.reshape(input_z, (len(obs), 1))
                else:
                    raise ValueError(
                        f"input_z length mismatch: got {len(input_z)} values for {len(obs)} observations"
                    )
            obs = np.concatenate([obs, z_col], axis=1)
        
        posteriors = []
        if bar:
            for i in trange(len(obs)):
                p = np.array(qphi.sample((n_samples,), x=torch.as_tensor(np.array([obs[i, :]]).astype(np.float32)).to(device), show_progress_bars=False).detach().to('cpu'))
                posteriors.append(p)
        else:
            for i in range(len(obs)):
                p = np.array(qphi.sample((n_samples,), x=torch.as_tensor(np.array([obs[i, :]]).astype(np.float32)).to(device), show_progress_bars=True).detach().to('cpu'))
                posteriors.append(p)

        return np.array(posteriors)        
        

    def get_posteriors_resolved(self, phot_arr, n_gal, n_samples=50, save=True, return_stats=True,sigma_arr=None, bar=True, input_z=None, device='cpu', sample_with='rejection'):
        """
        Generate posterior samples for resolved galaxy photometry.

        Parameters
        ----------
        phot_arr : np.ndarray
            Photometric data array (n_pixels, n_filters)
        n_gal : int
            Galaxy identifier for saving
        n_samples : int, optional
            Number of posterior samples (default: 50)
        save : bool, optional
            Whether to save results (default: True)
        return_stats : bool, optional
            Whether to return summary statistics (default: True)
        sigma_arr : np.ndarray, optional
            Photometric uncertainties
        bar : bool, optional
            Whether to show progress bar (default: True)
        input_z : float, optional
            Input redshift if not inferring
        device : str, optional
            Device for inference (default: 'cpu')
        sample_with : str, optional
            Posterior sampling backend. Use 'rejection' (default) or 'mcmc'.

        Returns
        -------
        Various arrays depending on options selected
            Posterior samples, summary statistics, and coordinate information
        """
        # Apply detection limits and convert to magnitudes
        if self.include_limit:
            for i in range(len(phot_arr[0, :])):
                # 1. Create mask BEFORE overwriting the array
                is_non_detect = phot_arr[:, i] < self.limits[i]
                
                # 2. Update photometry using the mask
                phot_arr[:, i] = np.where(
                    is_non_detect, 
                    99.0, # previously 0, but 99 is more standard for non-detections 
                    mag_conversion(phot_arr[:, i])
                )
                
                # 3. Update errors using the SAME mask
                if self.condition_sigma and sigma_arr is not None:
                    sigma_arr[:, i] = np.where(
                        is_non_detect, 
                        self.limits[i], 
                        np.abs(sigma_arr[:, i])
                    )
            mag_arr = phot_arr
        else:
            mag_arr = mag_conversion(phot_arr)
            coords_ok = np.where(~np.isnan(np.sum(mag_arr, axis=1)))[0]
            mag_arr = mag_arr[coords_ok, :]

        # Prepare full array with uncertainties if needed
        if self.condition_sigma:
            full_arr = np.zeros((len(phot_arr[:, 0]), len(phot_arr[0,:]), 2))
            full_arr[:, :, 0] = mag_arr
            
            for i in range(len(full_arr[:, 0, 0])):
                for j in range(len(full_arr[0, :, 0])):
                    if sigma_arr[i, j] == self.limits[j]:
                        full_arr[i, j, 1] = mag_conversion(self.limits[j], convert_to='mag')
                    else:
                        full_arr[i, j, 1] = (sigma_arr[i, j] * np.abs(-2.5 / (np.log(10) * phot_arr[i, j])))
            
            mag_arr = np.reshape(full_arr, (len(phot_arr[:, 0]), len(phot_arr[0, :]) * 2))
        
        self.gal = mag_arr

        # Load appropriate model
        if not self.infer_z_integrated:
            model_file = self.model_path + self.model_name
        else:
            model_file = self.model_path + 'integrated_z' + self.model_name
        
        with open(model_file, 'rb') as f:
            qphi = pickle.load(f)

        # sbi>=0.18 requires setting sampling backend in build_posterior(), not in sample()
        if sample_with != 'rejection':
            anpe_file = self.model_path + 'anpe_' + self.model_name
            try:
                with open(anpe_file, 'rb') as f:
                    anpe = pickle.load(f)
                qphi = anpe.build_posterior(sample_with=sample_with)
                print(f"Using posterior sampler backend: {sample_with}")
            except Exception as exc:
                print(
                    f"WARNING: could not rebuild posterior with sample_with='{sample_with}' "
                    f"from {anpe_file} ({exc}). Falling back to default sampler."
                )

        # Generate posteriors
        posteriors_full = self._get_posterior_obs(
            self.gal, qphi, n_samples=n_samples, bar=bar, 
            input_z=input_z, device=device
        )

        # Compute summary statistics if requested
        if return_stats:
            means = np.median(posteriors_full, axis=1)
            stds = np.std(posteriors_full, axis=1)
            modes = []
            for p in posteriors_full:
                mode = []
                for k in range(len(p[0, :])):
                    mode.append(stats.mode(np.round(p[:, k], 1)))
                modes.append(mode)

        # Save results if requested
        if save:
            np.save(f'post_gal_{n_gal}.npy', posteriors_full)
            if not self.include_limit:
                np.save(f'coords_gal_{n_gal}.npy', coords_ok)
            if return_stats:
                np.save(f'means_gal_{n_gal}.npy', means)
                np.save(f'stds_gal_{n_gal}.npy', stds)
                np.save(f'modes_gal_{n_gal}.npy', modes)

        # Return appropriate results
        if return_stats and not self.include_limit:
            return posteriors_full, means, stds, modes, coords_ok
        elif return_stats and self.include_limit:
            return posteriors_full, means, stds, modes
        elif self.include_limit and not return_stats:
            return posteriors_full
        else:
            return posteriors_full, coords_ok