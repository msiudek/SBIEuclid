from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

from sbipix import sbipix


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
N_SIM    = 100_000   # must match the atlas on disk
N_TEST   = 500
N_POSTERIOR = 200

# Set to True for a quick laptop smoke test (small training subset, few epochs)
SMOKE_TEST = False
if SMOKE_TEST:
    N_TRAIN_MAX = 5_000   # subsample after loading — atlas size stays 100k
    N_TEST      = 100
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
MODEL_NAME  = "model_euclid_v1.0.pkl"

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
print(f"    {len(sx.theta)} galaxies after physical range clip (logM: 4-13, logSFR: -30 to 5)")



# --------------------------------------------------
# SELECT TARGETS: logM + logSFR ONLY
# --------------------------------------------------
print("[3/5] Selecting parameters (logM, logSFR)...")
# theta column order is fixed: 0=logM*, 1=logM*_formed, 2=logSFR,
#                               3=tau, 4=t_i, 5=[M/H], 6=Av, 7=z
MASS_IDX = 0
SFR_IDX  = 2

sx.theta  = sx.theta[:, [MASS_IDX, SFR_IDX]]
sx.labels = ["logM", "logSFR"]

print(f"    Using columns: logM={MASS_IDX}, logSFR={SFR_IDX}")

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


sx.train(
    min_thetas=min_thetas,
    max_thetas=max_thetas,
    n_max=min(N_TRAIN_MAX, len(sx.theta)),
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

# posterior: (N_test, N_samples, 2)

theta_true = sx.theta[:posterior.shape[0]]

pred_mean = np.median(posterior, axis=1)


# --------------------------------------------------
# PLOT RESULTS
# --------------------------------------------------
n_params = posterior.shape[-1]
names = sx.labels[:n_params]
fig, axes = plt.subplots(1, max(n_params, 1), figsize=(5 * max(n_params, 1), 5))
axes = np.atleast_1d(axes)

for i in range(n_params):
    t = theta_true[:, i]
    p = pred_mean[:, i]
    axes[i].scatter(t, p, s=5, alpha=0.5)
    axes[i].plot([t.min(), t.max()], [t.min(), t.max()], "r--")
    axes[i].set_xlabel(f"True {names[i]}")
    axes[i].set_ylabel(f"Pred {names[i]}")
    axes[i].grid(alpha=0.3)

plt.suptitle(f"Recovery: {', '.join(names)}")
plt.tight_layout()
plt.savefig("recovery_logM_logSFR.png", dpi=150)
plt.show()

print("Saved: recovery_logM_logSFR.png")