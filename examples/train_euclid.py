import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

from sbipix import sbipix
from sbipix.plotting import plot_test_performance


def build_parser():
    p = argparse.ArgumentParser(description="Train SBIPIX model for Euclid photometry")
    p.add_argument(
        "--params",
        choices=["all", "mass_sfr"],
        default="mass_sfr",
        help="Parameter set to infer: all atlas parameters or only [logM, logMformed, logSFR]",
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
MODEL_NAME  = "model_euclid_v1.1.pkl"

NOISE_PREFIX = "north_2fwhm"


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
          (sx.theta[:, 2] > -30.0) & (sx.theta[:, 2] < 5.0)
sx.theta = sx.theta[phys_ok]
sx.mag   = sx.mag[phys_ok]
sx.obs   = sx.obs[phys_ok]
sx.n_simulation = len(sx.theta)
print(f"    {len(sx.theta)} galaxies after physical range clip (logM: 4-13, logSFR: -30 to 5)")



# --------------------------------------------------
# SELECT TARGETS
# --------------------------------------------------
print(f"[3/5] Selecting parameters ({args.params})...")

# theta column order is fixed: 0=logM*, 1=logM*_formed, 2=logSFR,
#                               3=tau, 4=t_i, 5=[M/H], 6=Av, 7=z
if args.params == "mass_sfr":
    PARAM_IDXS = [0, 1, 2]
    PARAM_NAMES = ["logM", "logMformed", "logSFR"]
    sx.theta = sx.theta[:, PARAM_IDXS]
    sx.labels = PARAM_NAMES
    # Important: with infer_z=False, sbipix.train() treats the last theta
    # column as known redshift and removes it from inferred targets.
    # For mass_sfr we want 3 inferred parameters, so enable infer_z mode.
    sx.infer_z = True
    print(f"    Using columns: {PARAM_IDXS} -> {PARAM_NAMES}")
    print("    infer_z=True for mass_sfr mode (infer all 3 selected parameters)")
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