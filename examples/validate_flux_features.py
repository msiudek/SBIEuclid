#!/usr/bin/env python
"""
Validate flux-space conditioning features per band.
Reports per-band statistics to diagnose if flux mode is suitable for training.
"""

import numpy as np
from pathlib import Path
from sbipix import sbipix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = PROJECT_ROOT / 'obs' / 'obs_properties'
LIB_DIR = PROJECT_ROOT / 'library'

# Configuration
ATLAS_NAME = 'atlas_obs_euclid_north_validate'
N_SIM = 100_000
NOISE_PREFIX = 'north_templfit'
SNR_DETECTION_THRESHOLD = 3.0

sx = sbipix()
sx.configure_filters(
    filter_list='filters_to_use.dat',
    filter_path=str(OBS_DIR),
    mean_sigma_file=f'mean_sigma_{NOISE_PREFIX}.npy',
    std_sigma_file=f'std_sigma_{NOISE_PREFIX}.npy',
    percentiles_file=f'percentiles_{NOISE_PREFIX}.npy',
    limits_file=f'background_noise_{NOISE_PREFIX}.npy',
    lam_eff_file=f'lam_eff_{NOISE_PREFIX}.npy',
)
sx.atlas_path = str(LIB_DIR) + '/'
sx.atlas_name = ATLAS_NAME
sx.n_simulation = N_SIM
sx.parametric = True
sx.both_masses = True
sx.infer_z = False
sx.include_limit = True
sx.include_sigma = True
sx.condition_sigma = True
sx.configure_noise_model(
    sigma_sampler='mag_lognormal',
    detection_model='hard',
    observation_space='flux',
)
sx.snr_threshold = SNR_DETECTION_THRESHOLD

print("Loading atlas and observational features...")
sx.load_simulation()
sx.load_obs_features()
sx.add_noise_nan_limit_all()

# Clean
mask = np.isfinite(sx.theta).all(axis=1) & np.isfinite(sx.mag).all(axis=(1, 2))
mask &= (sx.theta[:, 0] > 4.0) & (sx.theta[:, 0] < 13.0) & (sx.theta[:, 2] > -4.0) & (sx.theta[:, 2] < 3.0)
sx.theta = sx.theta[mask]
sx.mag = sx.mag[mask]
sx.obs = sx.obs[mask]
sx.n_simulation = len(sx.theta)

print(f"Sample size: {sx.n_simulation}")

# Extract raw and transformed features
flux_obs = np.asarray(sx.mag[:, :, 0], dtype=float)
sigma_flux = np.asarray(sx.mag[:, :, 1], dtype=float)
flux_true = 3631e6 * 10 ** (-0.4 * np.asarray(sx.obs, dtype=float))

# Get transformed features
obs_transformed = sx._get_conditioning_observations()
flux_tx = obs_transformed[:, 0::2]
sigma_tx = obs_transformed[:, 1::2]

# Get filter names
from sbipix.utils.sed_utils import load_filter_metadata
_FILTER_META = load_filter_metadata('filters_to_use.dat', filt_dir=str(OBS_DIR))
FILTER_SHORT = [m['short'] for m in _FILTER_META]

print("\n" + "="*120)
print("PER-BAND FLUX VALIDATION DIAGNOSTICS")
print("="*120)

# Header
print(f"{'Band':>12} | {'N_det':>8} | {'At_floor':>8} | {'At_cap':>8} | {'flux_tx':>8} | {'sigma_tx':>8} | {'Pull':>10} | {'Status':>8}")
print("-"*120)

issues = []

for fi, band_name in enumerate(FILTER_SHORT):
    flux_i = flux_obs[:, fi]
    sigma_i = sigma_flux[:, fi]
    flux_true_i = flux_true[:, fi]
    flux_tx_i = flux_tx[:, fi]
    sigma_tx_i = sigma_tx[:, fi]
    
    # Detection count (finite, not nan sentinel)
    detected = np.isfinite(flux_i) & (flux_i < 99.0)
    n_det = np.sum(detected)
    det_frac = 100 * n_det / len(flux_i) if len(flux_i) > 0 else 0
    
    # Fraction at sigma floor (noise_sigma_floor ~ 8e-3)
    at_floor = (np.abs(sigma_tx_i - np.log10(8e-3 / sx.limits[fi])) < 1e-3).sum()
    frac_floor = 100 * at_floor / len(sigma_i) if len(sigma_i) > 0 else 0
    
    # Fraction at sigma cap (noise_sigma_mag_max ~ 10.0)
    at_cap = (np.abs(sigma_tx_i - np.log10(10.0 / sx.limits[fi])) < 1e-3).sum()
    frac_cap = 100 * at_cap / len(sigma_i) if len(sigma_i) > 0 else 0
    
    # Transformed flux_tx std (measure of dynamic range after transform)
    flux_tx_finite = flux_tx_i[np.isfinite(flux_tx_i)]
    flux_tx_std = np.std(flux_tx_finite) if len(flux_tx_finite) > 0 else np.nan
    
    # Transformed sigma_tx std (measure of error diversity)
    sigma_tx_finite = sigma_tx_i[np.isfinite(sigma_tx_i)]
    sigma_tx_std = np.std(sigma_tx_finite) if len(sigma_tx_finite) > 0 else np.nan
    
    # Pull stats (noise calibration)
    if detected.sum() > 0:
        pull = (flux_obs[detected, fi] - flux_true_i[detected]) / np.maximum(sigma_i[detected], 1e-12)
        pull_finite = pull[np.isfinite(pull)]
        if len(pull_finite) > 0:
            pull_mean = np.mean(pull_finite)
            pull_std = np.std(pull_finite)
            pull_status = f"{pull_mean:+.3f}±{pull_std:.2f}"
        else:
            pull_status = "no pulls"
    else:
        pull_status = "not_det"
    
    # Status check
    status = "OK"
    if frac_floor > 50:
        status = "WARN"
        issues.append(f"Band {band_name}: {frac_floor:.1f}% at sigma floor (may be over-censored)")
    if frac_cap > 20:
        status = "WARN"
        issues.append(f"Band {band_name}: {frac_cap:.1f}% at sigma cap (noise model saturated)")
    if det_frac < 5:
        status = "BAD"
        issues.append(f"Band {band_name}: only {det_frac:.1f}% detected (too faint for training)")
    if np.isnan(flux_tx_std) or flux_tx_std > 10:
        status = "BAD"
        issues.append(f"Band {band_name}: flux_tx_std={flux_tx_std:.2f} (bad transformed range)")
    
    print(f"{band_name:>12} | {n_det:>8d} | {frac_floor:>7.1f}% | {frac_cap:>7.1f}% | {flux_tx_std:>7.2f}  | {sigma_tx_std:>7.2f}  | {pull_status:>10} | {status:>8}")

print("="*120)

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  ⚠️  {issue}")
    print("\nRECOMMENDATION: Review problematic bands before flux-mode training.")
else:
    print("\n✅ All bands PASS flux validation. Flux-space training is suitable.")

print("\nSUMMARY STATS (across all bands):")
det_frac_all = 100 * (flux_obs < 99.0).sum() / flux_obs.size
floor_frac_all = 100 * (np.abs(sigma_tx - np.log10(8e-3 / sx.limits[np.newaxis, :])) < 1e-3).sum() / sigma_tx.size
cap_frac_all = 100 * (np.abs(sigma_tx - np.log10(10.0 / sx.limits[np.newaxis, :])) < 1e-3).sum() / sigma_tx.size

print(f"  Detection fraction: {det_frac_all:.1f}%")
print(f"  At sigma floor: {floor_frac_all:.1f}%")
print(f"  At sigma cap: {cap_frac_all:.1f}%")

# Overall verdict
if not issues and det_frac_all > 10 and floor_frac_all < 50 and cap_frac_all < 20:
    print("\n🟢 VERDICT: Flux-space training is RECOMMENDED for this sample.")
elif not issues:
    print("\n🟡 VERDICT: Flux-space training is ACCEPTABLE but monitor training stability.")
else:
    print("\n🔴 VERDICT: Consider reverting to magnitude-space until issues are resolved.")
