import argparse
from pathlib import Path
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
        "--mock-match",
        choices=["none", "vis_yj2d"],
        default="vis_yj2d",
        help="Resample mocks to match real observed VIS×(Y-J) prior (default: vis_yj2d)",
    )
    return p


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
N_SIM    = 100_000   # must match the atlas on disk
N_TEST   = 250
N_POSTERIOR = 200

args = build_parser().parse_args()
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

ATLAS_NAME  = "atlas_obs_euclid_north_validate"
MODEL_NAME  = "model_euclid_v1.3.pkl"

NOISE_PREFIX = "north_2fwhm"
NONDET_MAG = 99.0
SNR_DETECTION_THRESHOLD = 2.0
MAG_BRIGHT = 16.0
MAG_FAINT = 30.0
PATCH_ID = 98

_FILTER_META = load_filter_metadata("filters_to_use.dat", filt_dir=str(OBS_DIR))
FILTER_SHORT = [m["short"] for m in _FILTER_META]
FILTER_COL_STEMS = [m["col_stem"] for m in _FILTER_META]


def load_real_mag_for_mock_match(aperture):
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
        fcol = f"flux_{stem}_{aperture}_aper"
        ecol = f"fluxerr_{stem}_{aperture}_aper"
        if fcol not in cat.colnames:
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

    real_hist, _, _ = np.histogram2d(real_vis[real_ok], real_yj[real_ok], bins=(bins_mag, bins_color))
    mock_hist, _, _ = np.histogram2d(mock_vis[mock_ok], mock_yj[mock_ok], bins=(bins_mag, bins_color))

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
sx.model_name = MODEL_NAME
sx.n_simulation = N_SIM

sx.parametric = True
sx.both_masses = True
sx.infer_z = False

sx.include_limit = True
sx.include_sigma = True
sx.condition_sigma = True

sx.configure_noise_model(
    sigma_sampler="mag_lognormal",
    detection_model="hard",
)


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
    print(f"    Applying mock matching ({args.mock_match})...")
    aperture = NOISE_PREFIX.split("_", 1)[1] if "_" in NOISE_PREFIX else "2fwhm"
    real_mag = load_real_mag_for_mock_match(aperture=aperture)
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



# --------------------------------------------------
# SELECT TARGETS
# --------------------------------------------------
print(f"[3/5] Selecting parameters ({args.params})...")

# theta column order is fixed: 0=logM*, 1=logM*_formed, 2=logSFR,
#                               3=tau, 4=t_i, 5=[M/H], 6=Av, 7=z
if args.params == "mass_sfr":
    PARAM_IDXS = [0, 2]
    PARAM_NAMES = ["logM", "logSFR"]
    sx.theta = sx.theta[:, PARAM_IDXS]
    sx.labels = PARAM_NAMES
    # Important: with infer_z=False, sbipix.train() treats the last theta
    # column as known redshift and removes it from inferred targets.
    # For mass_sfr we want all selected parameters inferred, so enable infer_z mode.
    sx.infer_z = True
    print(f"    Using columns: {PARAM_IDXS} -> {PARAM_NAMES}")
    print("    infer_z=True for mass_sfr mode (infer all selected parameters)")
else:
    print(f"    Using all parameters: {sx.theta.shape[1]} dimensions")

sx.n_simulation = len(sx.theta)

# Parameter bounds for normalization
max_thetas = np.max(sx.theta, axis=0)
min_thetas = np.min(sx.theta, axis=0)

print("Parameter bounds:")
for i, name in enumerate(sx.labels):
    print(f"   - {name}: [{min_thetas[i]:.2f}, {max_thetas[i]:.2f}]")

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