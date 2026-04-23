import argparse
from pathlib import Path
from typing import Any
import re
import numpy as np

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

from sbipix import sbipix
from sbipix.plotting import plot_test_performance, plot_training_history
from sbipix.utils.sed_utils import flux_ujy_to_mag, load_filter_metadata


def build_parser():
    p = argparse.ArgumentParser(description="Train SBIPIX model for Euclid photometry")
    p.add_argument(
        "--params",
        choices=["all", "mass_sfr"],
        default="mass_sfr",
        help="Parameter set to infer: all atlas parameters or only [logM, logSFR]",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick local check (train on 5k objects, 20 epochs)",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=250,
        help="Number of galaxies used in test-time inference/performance (default: 250)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse existing model file at model_path/model_name",
    )
    p.add_argument(
        "--sanity-check-only",
        action="store_true",
        help="Run preprocessing + flux sanity diagnostics, then exit before training",
    )
    p.add_argument(
        "--mock-match",
        choices=["none", "vis_yj2d"],
        default="vis_yj2d",
        help="Resample mocks to match real observed VIS×(Y-J) prior (default: vis_yj2d)",
    )
    p.add_argument(
        "--z-mode",
        choices=["condition", "infer"],
        default="condition",
        help=(
            "How redshift is handled. "
            "condition: use z as known conditioning input (recommended for catalog-z inference); "
            "infer: infer z from photometry."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output model filename (default auto-selected from params and z-mode)",
    )
    p.add_argument(
        "--atlas-name",
        type=str,
        default="atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas",
        help="Atlas filename in library/ to use for training (default: 100k north validate atlas)",
    )
    p.add_argument(
        "--n-sim",
        type=int,
        default=100_000,
        help="Number of simulations represented by the atlas (default: 100000)",
    )
    p.add_argument(
        "--phot-type",
        choices=["2fwhm", "3fwhm", "templfit"],
        default="templfit",
        help=(
            "Photometry type used for noise model and mock matching. "
            "'templfit' uses template-fit fluxes (flux_{stem}_templfit; VIS: flux_vis_psf). "
            "'2fwhm'/'3fwhm' use fixed-aperture fluxes. Default: templfit"
        ),
    )
    p.add_argument(
        "--sigma-sampler",
        choices=["empirical", "truncnorm", "mag_lognormal"],
        default="mag_lognormal",
        help="Sigma sampling mode for mock-noise injection (default: mag_lognormal)",
    )
    p.add_argument(
        "--detection-model",
        choices=["hard", "probabilistic"],
        default="hard",
        help="Detection model used after noise injection (default: hard)",
    )
    p.add_argument(
        "--observation-space",
        choices=["mag", "flux"],
        default="mag",
        help=(
            "Feature space passed to SBI: 'mag' keeps legacy magnitude+sigma features; "
            "'flux' uses noisy flux+sigma_flux and keeps negative noisy realizations."
        ),
    )
    return p


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
args = build_parser().parse_args()
N_SIM    = args.n_sim
N_TEST   = 250
N_POSTERIOR = 200

N_TEST = args.n_test

# Set to True for a quick laptop smoke test (small training subset, few epochs)
SMOKE_TEST = args.smoke_test
if SMOKE_TEST:
    N_TRAIN_MAX = 5_000   # subsample after loading — atlas size stays 100k
    N_POSTERIOR = 50
else:
    N_TRAIN_MAX = N_SIM

print(f"Device: {_DEVICE}")
if _DEVICE == "cpu":
    print("WARNING: no GPU detected — training will be slow (~30-90 min for 100k galaxies).")
    print("         Consider running on the server, or set SMOKE_TEST=True for a quick check.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = PROJECT_ROOT / "obs" / "obs_properties"
LIB_DIR = PROJECT_ROOT / "library"

ATLAS_NPARAM = 2


def normalize_atlas_name(atlas_name: str, n_sim: int, n_param: int) -> str:
    """Return dense_basis atlas stem from either stem or full .dbatlas filename."""
    value = str(atlas_name).strip()
    if value.endswith(".dbatlas"):
        value = value[: -len(".dbatlas")]

    suffix = f"_{int(n_sim)}_Nparam_{int(n_param)}"
    if value.endswith(suffix):
        value = value[: -len(suffix)]

    # Safety net for unusual duplicated endings provided by user input.
    value = re.sub(r"_\d+_Nparam_\d+$", "", value)
    return value


ATLAS_NAME = normalize_atlas_name(args.atlas_name, N_SIM, ATLAS_NPARAM)

# Photometry column helper (mirrors learn_obs_noise_from_survey.py)
def build_phot_col(stem, phot_type, err=False):
    """Return flux/fluxerr column name for a filter stem and photometry type."""
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"
    return f"{prefix}_{stem}_{phot_type}_aper"

NOISE_PREFIX = f"north_{args.phot_type}"
NONDET_MAG = 99.0
SNR_DETECTION_THRESHOLD = 3.0
MAG_BRIGHT = 16.0
MAG_FAINT = 30.0
PATCH_ID = 98
MATCHED_CATALOG = OBS_DIR / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"

_FILTER_META = load_filter_metadata("filters_to_use.dat", filt_dir=str(OBS_DIR))
FILTER_SHORT = [m["short"] for m in _FILTER_META]
FILTER_COL_STEMS = [m["col_stem"] for m in _FILTER_META]


def load_real_mag_for_mock_match(phot_type):
    """Load real detected magnitudes (filter-major arrays) for prior matching."""
    from astropy.table import Table

    fits_path = OBS_DIR / "COSMOS_DEEP.fits"
    cat = Table.read(fits_path)

    patch_col = cat["patch_id_list"]
    try:
        mask = patch_col == int(PATCH_ID)
    except (ValueError, TypeError):
        mask = np.zeros(len(cat), dtype=bool)
    str_mask = np.array([str(v).strip() == str(PATCH_ID) for v in patch_col])
    cat = cat[mask | str_mask]

    n_filt = len(FILTER_COL_STEMS)
    n_gal = len(cat)
    real_mag = np.full((n_filt, n_gal), np.nan)

    for fi, stem in enumerate(FILTER_COL_STEMS):
        fcol = build_phot_col(stem, phot_type, err=False)
        ecol = build_phot_col(stem, phot_type, err=True)
        if fcol not in cat.colnames:
            print(f"  WARNING: column '{fcol}' not found in COSMOS_DEEP.fits — skipping filter {stem}")
            continue

        flux = np.asarray(cat[fcol], dtype=float)
        err = np.asarray(cat[ecol], dtype=float) if ecol in cat.colnames else np.full(n_gal, np.nan)
        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        snr = np.where(valid, flux / err, np.nan)
        detected = valid & np.isfinite(snr) & (snr >= SNR_DETECTION_THRESHOLD) & (flux > 0)
        real_mag[fi] = np.where(detected, flux_ujy_to_mag(flux), np.nan)

    return real_mag


def compute_mock_match_weights(real_mag, mock_mag):
    """Compute 2D VIS×(Y-J) histogram ratio weights for mock samples."""
    n_mock = mock_mag.shape[1]
    weights = np.zeros(n_mock, dtype=float)

    vis_idx = FILTER_SHORT.index("VIS")
    y_idx = FILTER_SHORT.index("NISP-Y")
    j_idx = FILTER_SHORT.index("NISP-J")

    real_vis = real_mag[vis_idx]
    real_yj = real_mag[y_idx] - real_mag[j_idx]
    real_ok = np.isfinite(real_vis) & np.isfinite(real_yj)

    mock_vis = mock_mag[vis_idx]
    mock_yj = mock_mag[y_idx] - mock_mag[j_idx]
    mock_ok = np.isfinite(mock_vis) & (mock_vis < NONDET_MAG - 0.5) & np.isfinite(mock_yj)

    bins_mag = np.linspace(MAG_BRIGHT, MAG_FAINT, 25)
    if np.any(real_ok):
        p1, p99 = np.nanpercentile(real_yj[real_ok], [1, 99])
        bins_color = np.linspace(p1 - 0.5, p99 + 0.5, 25)
    else:
        bins_color = np.linspace(-2.0, 4.0, 25)

    n_real = int(real_ok.sum())
    n_mock_ok = int(mock_ok.sum())
    if n_real == 0 or n_mock_ok == 0:
        return weights, "mock matching skipped: no valid VIS+Y-J objects"

    bins_2d: Any = (bins_mag, bins_color)
    real_hist, _, _ = np.histogram2d(real_vis[real_ok], real_yj[real_ok], bins=bins_2d)
    mock_hist, _, _ = np.histogram2d(mock_vis[mock_ok], mock_yj[mock_ok], bins=bins_2d)

    real_pdf = real_hist / n_real
    mock_pdf = mock_hist / n_mock_ok
    ratio = real_pdf / (mock_pdf + 1e-6)
    ratio = np.clip(ratio, 0, 10)

    ix = np.digitize(mock_vis, bins_mag) - 1
    iy = np.digitize(mock_yj, bins_color) - 1
    in_range = mock_ok & (ix >= 0) & (ix < ratio.shape[0]) & (iy >= 0) & (iy < ratio.shape[1])
    weights[in_range] = ratio[ix[in_range], iy[in_range]]

    return weights, (
        f"mock matching: 2D VIS×(Y-J) histogram 24×24 bins "
        f"(real n={n_real}, mock detected n={n_mock_ok})"
    )


def draw_resample_indices(weights, n_out, seed=0):
    """Draw bootstrap indices from positive weights."""
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None, "mock matching skipped: all weights are zero"

    probs = weights[valid] / weights[valid].sum()
    source_idx = np.where(valid)[0]
    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(source_idx, size=n_out, replace=True, p=probs)
    eff_n = (weights[valid].sum() ** 2) / np.sum(weights[valid] ** 2)
    return draw_idx, f"resampled weighted mock catalogue (effective N ≈ {eff_n:.0f})"


# --------------------------------------------------
# INIT MODEL
# --------------------------------------------------
sx = sbipix()

sx.configure_filters(
    filter_list="filters_to_use.dat",
    filter_path=str(OBS_DIR),
    mean_sigma_file=f"mean_sigma_{NOISE_PREFIX}.npy",
    std_sigma_file=f"std_sigma_{NOISE_PREFIX}.npy",
    percentiles_file=f"percentiles_{NOISE_PREFIX}.npy",
    limits_file=f"background_noise_{NOISE_PREFIX}.npy",
    lam_eff_file=f"lam_eff_{NOISE_PREFIX}.npy",
)

sx.atlas_path = str(LIB_DIR) + "/"
sx.model_path = str(LIB_DIR) + "/"

sx.atlas_name = ATLAS_NAME
print(
    "Atlas resolved for dense_basis: "
    f"stem='{sx.atlas_name}', file='{sx.atlas_name}_{N_SIM}_Nparam_{ATLAS_NPARAM}.dbatlas'"
)
if args.model_name is not None:
    sx.model_name = args.model_name
else:
    if args.params == "mass_sfr" and args.z_mode == "condition":
        sx.model_name = "model_euclid_v1.5_mass_sfr_zcond.pkl"
    elif args.params == "mass_sfr" and args.z_mode == "infer":
        sx.model_name = "model_euclid_v1.5_mass_sfr_zinfer.pkl"
    elif args.params == "all" and args.z_mode == "condition":
        sx.model_name = "model_euclid_v1.5_all_zcond.pkl"
    else:
        sx.model_name = "model_euclid_v1.5_all_zinfer.pkl"
sx.n_simulation = N_SIM

sx.parametric = True
sx.both_masses = True
sx.infer_z = False

sx.include_limit = True
sx.include_sigma = True
sx.condition_sigma = True

sx.configure_noise_model(
    sigma_sampler=args.sigma_sampler,
    detection_model=args.detection_model,
    observation_space=args.observation_space,
)
sx.snr_threshold = SNR_DETECTION_THRESHOLD

print(
    f"Noise model: sigma_sampler={sx.noise_sigma_sampler}, "
    f"detection_model={sx.noise_detection_model}, "
    f"observation_space={sx.noise_observation_space}"
)
print(f"Noise SNR detection threshold: {sx.snr_threshold}")


# --------------------------------------------------
# LOAD ATLAS
# --------------------------------------------------
print("[1/5] Loading atlas...")
sx.load_simulation()


# --------------------------------------------------
# ADD REALISM
# --------------------------------------------------
print("[2/5] Adding observational realism...")
sx.load_obs_features()
sx.add_noise_nan_limit_all()

# clean NaNs
ok = np.isfinite(np.sum(sx.theta, axis=1))
sx.theta = sx.theta[ok]
sx.mag   = sx.mag[ok]
sx.obs   = sx.obs[ok]

print(f"    {len(sx.theta)} valid galaxies (after NaN cleaning)")

# Clip to physical parameter ranges
phys_ok = (sx.theta[:, 0] > 4.0) & (sx.theta[:, 0] < 13.0) & \
          (sx.theta[:, 2] > -4.0) & (sx.theta[:, 2] < 3.0)
sx.theta = sx.theta[phys_ok]
sx.mag   = sx.mag[phys_ok]
sx.obs   = sx.obs[phys_ok]
sx.n_simulation = len(sx.theta)
print(f"    {len(sx.theta)} galaxies after physical range clip (logM: 4-13, logSFR: -4 to 3)")

if args.mock_match != "none":
    if args.observation_space == "flux":
        print(
            "    Flux-space mode: disabling mag-based mock matching to keep "
            "a strict flux-only pipeline."
        )
        args.mock_match = "none"

if args.mock_match != "none":
    print(f"    Applying mock matching ({args.mock_match})...")
    real_mag = load_real_mag_for_mock_match(phot_type=args.phot_type)
    if args.observation_space == "flux":
        mock_flux = sx.mag[:, :, 0].T
        with np.errstate(divide='ignore', invalid='ignore'):
            mock_mag = np.where(mock_flux > 0, flux_ujy_to_mag(mock_flux), np.nan)
    else:
        mock_mag = sx.mag[:, :, 0].T

    mock_weights, match_msg = compute_mock_match_weights(real_mag, mock_mag)
    print(f"      {match_msg}")
    print(f"      non-zero weights: {(mock_weights > 0).sum()} / {mock_weights.size}")

    draw_idx, resample_msg = draw_resample_indices(mock_weights, n_out=len(sx.theta), seed=0)
    print(f"      {resample_msg}")
    if draw_idx is not None:
        sx.theta = sx.theta[draw_idx]
        sx.mag = sx.mag[draw_idx]
        sx.obs = sx.obs[draw_idx]
        sx.n_simulation = len(sx.theta)
        print(f"      post-match n_simulation: {sx.n_simulation}")
else:
    print("    Mock matching disabled (--mock-match none)")

if args.observation_space == "flux":
    print("    Flux-space strict mode: training features are [flux_i, sigma_flux_i].")
    flux_obs = np.asarray(sx.mag[:, :, 0], dtype=float)
    sigma_flux = np.asarray(sx.mag[:, :, 1], dtype=float)
    flux_true = 3631e6 * 10 ** (-0.4 * np.asarray(sx.obs, dtype=float))

    flux_finite = flux_obs[np.isfinite(flux_obs)]
    sigma_finite = sigma_flux[np.isfinite(sigma_flux) & (sigma_flux > 0)]
    if flux_finite.size > 0:
        p1, p50, p99 = np.percentile(flux_finite, [1, 50, 99])
        print(f"    np.percentile(flux, [1,50,99]) = [{p1:.6g}, {p50:.6g}, {p99:.6g}]")
    else:
        print("    np.percentile(flux, [1,50,99]) = [nan, nan, nan] (no finite flux values)")

    if sigma_finite.size > 0:
        s1, s50, s99 = np.percentile(sigma_finite, [1, 50, 99])
        print(f"    np.percentile(sigma_flux, [1,50,99]) = [{s1:.6g}, {s50:.6g}, {s99:.6g}]")
    else:
        print("    np.percentile(sigma_flux, [1,50,99]) = [nan, nan, nan] (no positive finite sigma)")

    pull = (flux_obs - flux_true) / np.where(sigma_flux > 0, sigma_flux, np.nan)
    pull = pull[np.isfinite(pull)]
    if pull.size > 0:
        print(f"    np.mean((flux_obs - flux_true)/sigma_flux) = {np.mean(pull):+.4f}")
        print(f"    np.std((flux_obs - flux_true)/sigma_flux)  = {np.std(pull):.4f}")
    else:
        print("    pull stats unavailable (no finite pull values)")

    if args.sanity_check_only:
        print("    --sanity-check-only set: exiting before training.")
        raise SystemExit(0)




# --------------------------------------------------
# SELECT TARGETS
# --------------------------------------------------
print(f"[3/5] Selecting parameters ({args.params}, z-mode={args.z_mode})...")

# theta column order is fixed: 0=logM*, 1=logM*_formed, 2=logSFR,
#                               3=tau, 4=t_i, 5=[M/H], 6=Av, 7=z
if args.params == "mass_sfr":
    if args.z_mode == "condition":
        PARAM_IDXS = [0, 2, 7]
    else:
        PARAM_IDXS = [0, 2]
    PARAM_NAMES = ["logM", "logSFR"]
    sx.theta = sx.theta[:, PARAM_IDXS]
    sx.labels = PARAM_NAMES
    # infer_z=False: treat last theta column as known redshift conditioning input.
    # infer_z=True: infer all selected parameters directly from photometry.
    sx.infer_z = args.z_mode == "infer"
    print(f"    Using columns: {PARAM_IDXS} -> {PARAM_NAMES}")
    if sx.infer_z:
        print("    Redshift handling: infer z from photometry (no catalog-z conditioning)")
    else:
        print("    Redshift handling: condition on known z (last theta column)")
else:
    sx.infer_z = args.z_mode == "infer"
    print(f"    Using all parameters: {sx.theta.shape[1]} dimensions")
    if sx.infer_z:
        print("    Redshift handling: infer z as part of target vector")
    else:
        print("    Redshift handling: condition on known z (theta[:, -1])")

sx.n_simulation = len(sx.theta)

# Parameter bounds for normalization
max_thetas = np.max(sx.theta, axis=0)
min_thetas = np.min(sx.theta, axis=0)

print("Parameter bounds:")
for i, name in enumerate(sx.labels):
    print(f"   - {name}: [{min_thetas[i]:.2f}, {max_thetas[i]:.2f}]")
print(f"Model output file: {sx.model_path}{sx.model_name}")

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("[4/5] Training...")

n_train_use = min(N_TRAIN_MAX, len(sx.theta))
print(f"    Training samples used (n_max): {n_train_use}")

if args.skip_train:
    model_file = Path(sx.model_path) / sx.model_name
    if not model_file.exists():
        raise FileNotFoundError(
            f"--skip-train requested but model file does not exist: {model_file}"
        )
    print(f"    Skipping training; reusing existing model: {model_file}")
else:
    sx.train(
        min_thetas=min_thetas,
        max_thetas=max_thetas,
        n_max=n_train_use,
        nblocks=4,
        nhidden=128,
        epochs_max=20 if SMOKE_TEST else 200,
    )


# --------------------------------------------------
# TEST
# --------------------------------------------------
print("[5/5] Testing recovery...")

posterior = sx.test_performance(
    n_test=min(N_TEST, len(sx.theta)),
    n_samples=N_POSTERIOR,
    return_posterior=True,
)

print("Performance test complete!")
print(f"   - Tested on {posterior.shape[0]} galaxies")
print(f"   - Posterior shape: {posterior.shape}")

expected_params = len(sx.labels)
actual_params = posterior.shape[-1]
if actual_params != expected_params:
    msg = (
        f"Posterior dimension mismatch: expected {expected_params} from labels {sx.labels}, "
        f"got {actual_params}."
    )
    if args.skip_train:
        raise RuntimeError(
            msg + " Likely reusing an older incompatible model; rerun without --skip-train."
        )
    else:
        print(f"WARNING: {msg}")

# posterior: (N_test, N_samples, n_params)

# --------------------------------------------------
# PLOT RESULTS
# --------------------------------------------------
n_params_plot = min(posterior.shape[-1], sx.theta.shape[1], len(sx.labels))
plot_name = f"test_performance_{args.params}_"
plot_test_performance(
    sx,
    n_test=min(N_TEST, len(sx.theta)),
    n_theta=n_params_plot,
    save=True,
    name=plot_name,
)
print(f"Saved: sbi-logs/{plot_name}*.png")

history_name = f"./sbi-logs/training_history_{args.params}.png"
history_file = plot_training_history(
    sx,
    save=True,
    filename=history_name,
)
if history_file:
    print(f"Saved: {history_file}")