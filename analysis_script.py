import numpy as np
import h5py
from scipy.stats import spearmanr

# Load inference results
res = np.load('sbi-logs/inference_cosmosweb_v1.0_50k_snr3/inference_results.npz', allow_pickle=True)
logM_sbi = res['logM_sbi']
logSFR_sbi = res['logSFR_sbi']
logM_cosmosweb = res['logM_cosmosweb']
logSFR_cosmosweb = res['logSFR_cosmosweb']
z = res['z']
n_bands = res['n_bands']

mask = np.isfinite(logM_sbi) & np.isfinite(logM_cosmosweb)
logM_sbi = logM_sbi[mask]
logSFR_sbi = logSFR_sbi[mask]
logM_cosmosweb = logM_cosmosweb[mask]
logSFR_cosmosweb = logSFR_cosmosweb[mask]
z = z[mask]
n_bands = n_bands[mask]

delta_logM = logM_sbi - logM_cosmosweb
delta_logSFR = logSFR_sbi - logSFR_cosmosweb

print(f"Overall median bias: logM={np.median(delta_logM):.3f}, logSFR={np.median(delta_logSFR):.3f}")

# logM bias by z bins
z_bins = [0, 0.5, 1, 2, 10]
for i in range(len(z_bins)-1):
    m = (z >= z_bins[i]) & (z < z_bins[i+1])
    if np.sum(m) > 0:
        print(f"logM bias z [{z_bins[i]},{z_bins[i+1]}): {np.median(delta_logM[m]):.3f} (N={np.sum(m)})")

# logM bias by n_bands bins
nb_bins = [7, 8, 9, 11]
for i in range(len(nb_bins)-1):
    m = (n_bands >= nb_bins[i]) & (n_bands < nb_bins[i+1])
    if np.sum(m) > 0:
        print(f"logM bias n_bands [{nb_bins[i]},{nb_bins[i+1]}): {np.median(delta_logM[m]):.3f} (N={np.sum(m)})")

# Max and 99th percentile
print(f"Max logM_sbi: {np.max(logM_sbi):.3f}, 99th: {np.percentile(logM_sbi, 99):.3f}")
print(f"Max reference logM: {np.max(logM_cosmosweb):.3f}, 99th: {np.percentile(logM_cosmosweb, 99):.3f}")

# Fractions > threshold
for thr in [11.2, 11.3, 11.4]:
    f_sbi = np.mean(logM_sbi > thr)
    f_ref = np.mean(logM_cosmosweb > thr)
    print(f"Fraction > {thr}: SBI={f_sbi:.4f}, Ref={f_ref:.4f}")

# Correlation
print(f"Corr(delta_logM, z): {spearmanr(delta_logM, z)[0]:.3f}")
print(f"Corr(delta_logM, n_bands): {spearmanr(delta_logM, n_bands)[0]:.3f}")

# Galaxies with cosmosweb logM > 11.3
m_heavy = logM_cosmosweb > 11.3
if np.sum(m_heavy) > 0:
    print(f"For heavy galaxies (Ref > 11.3, N={np.sum(m_heavy)}):")
    print(f"  Median inferred logM: {np.median(logM_sbi[m_heavy]):.3f}")
    print(f"  Median bias: {np.median(delta_logM[m_heavy]):.3f}")
else:
    print("No galaxies with reference logM > 11.3 found.")

# Load atlas mstar to check range
with h5py.File('library/atlas_obs_euclid_north_validate_50000_Nparam_2.dbatlas', 'r') as f:
    # hickle stores keys as strings with quotes in some versions
    key = '"mstar"' if '"mstar"' in f['data'] else 'mstar'
    atlas_mstar = f['data'][key][()]
    print(f"Atlas mstar max: {np.max(atlas_mstar):.3f}, 99th: {np.percentile(atlas_mstar, 99):.3f}")

# Check for posteriors
if 'posteriors' in res.files:
    posteriors = res['posteriors']
    if len(posteriors.shape) == 3:
        logM_samples = posteriors[:, :, 0]
        print(f"Max posterior sample: {np.max(logM_samples):.3f}")
        per_gal_max = np.max(logM_samples, axis=1)
        print(f"Median of per-galaxy posterior maxima: {np.median(per_gal_max):.3f}")
