#!/usr/bin/env python
"""
Diagnose how non-detection encoding biases mass estimates across redshift.

KEY QUESTION: When objects are non-detected (mag=99.0), do we incorrectly assign
errors that bias stellar mass estimation, especially at high-z where more objects
are non-detected?
"""

import numpy as np
from pathlib import Path
from sbipix import sbipix, mag_conversion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = PROJECT_ROOT / 'obs' / 'obs_properties'
LIB_DIR = PROJECT_ROOT / 'library'

# Configuration
ATLAS_NAME = 'atlas_obs_euclid_north_validate'
N_SIM = 100_000
NOISE_PREFIX = 'north_templfit'

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
    observation_space='mag',  # Use magnitude mode to see non-detection encoding directly
)
sx.snr_threshold = 3.0

print("Loading atlas and observational features...")
sx.load_simulation()
sx.load_obs_features()

# Before adding noise - original properties
z_true = sx.theta[:, 2]
ms_true = sx.theta[:, 0]
flux_true_all = sx.obs.copy()

print("Adding observational noise...")
sx.add_noise_nan_limit_all()

# After adding noise - check non-detections
mag_obs = sx.mag  # shape: (n_objects, n_filters, 2) where last dim is [mag, mag_err]
n_nondet = np.sum(mag_obs[:, :, 0] == 99.0)
total = mag_obs.shape[0] * mag_obs.shape[1]

print(f"\nNon-detections: {n_nondet} / {total} ({100*n_nondet/total:.2f}%)")

# Clean sample
mask = np.isfinite(sx.theta).all(axis=1) & np.isfinite(sx.mag).all(axis=(1, 2))
mask &= (sx.theta[:, 0] > 4.0) & (sx.theta[:, 0] < 13.0) & (sx.theta[:, 2] > -4.0) & (sx.theta[:, 2] < 3.0)

z_clean = z_true[mask]
ms_clean = ms_true[mask]
mag_clean = mag_obs[mask]
flux_clean = flux_true_all[mask]

print(f"Clean sample: {len(z_clean)} objects")

# Get filter names
from sbipix.utils.sed_utils import load_filter_metadata
_FILTER_META = load_filter_metadata('filters_to_use.dat', filt_dir=str(OBS_DIR))
FILTER_SHORT = [m['short'] for m in _FILTER_META]

print("\n" + "="*100)
print("NON-DETECTION BIAS ACROSS REDSHIFT")
print("="*100)

# Bin by redshift
z_bins = np.linspace(-4, 3, 15)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

print(f"\n{'z_bin':>10} | {'N':>6} | {'frac_nondet':>12} | {'frac_nondet_min':>15} | {'frac_nondet_max':>15} | {'frac_nondet_red':>15} | {'MS_offset':>10}")
print("-"*120)

# Red sequence colors (simple heuristic: rest-frame u-r color)
# For now just use stellar mass as proxy

z_offsets = []
for i in range(len(z_bins) - 1):
    mask_bin = (z_clean >= z_bins[i]) & (z_clean < z_bins[i + 1])
    if mask_bin.sum() < 10:
        continue
    
    z_avg = z_centers[i]
    n_bin = mask_bin.sum()
    
    # Fraction non-detected per band
    mag_bin = mag_clean[mask_bin]
    nondet_per_band = (mag_bin[:, :, 0] == 99.0).sum(axis=0) / n_bin
    frac_nondet_avg = nondet_per_band.mean()
    frac_nondet_min = nondet_per_band.min()
    frac_nondet_max = nondet_per_band.max()
    
    # Red objects (high stellar mass)
    ms_bin = ms_clean[mask_bin]
    red_mask = ms_bin > np.median(ms_bin)
    if red_mask.sum() > 5:
        mag_red = mag_bin[red_mask]
        nondet_red = (mag_red[:, :, 0] == 99.0).sum(axis=0) / red_mask.sum()
        frac_nondet_red = nondet_red.mean()
    else:
        frac_nondet_red = np.nan
    
    # What errors are assigned to non-detections?
    err_at_nondet = mag_bin[mag_bin[:, :, 0] == 99.0, 1]
    if len(err_at_nondet) > 0:
        err_nondet_avg = np.mean(err_at_nondet)
    else:
        err_nondet_avg = np.nan
    
    # Simple mass offset (between high and low mass at this z)
    ms_high = ms_bin[ms_bin > np.percentile(ms_bin, 75)]
    ms_low = ms_bin[ms_bin < np.percentile(ms_bin, 25)]
    if len(ms_high) > 2 and len(ms_low) > 2:
        ms_offset = np.median(ms_high) - np.median(ms_low)
    else:
        ms_offset = np.nan
    
    print(f"  {z_avg:>9.2f} | {n_bin:>6d} | {frac_nondet_avg:>11.1%} | {frac_nondet_min:>14.1%} | {frac_nondet_max:>14.1%} | {frac_nondet_red:>14.1%} | {err_nondet_avg:>9.2f}")
    
    z_offsets.append((z_avg, frac_nondet_avg, ms_offset))

print("\n" + "="*100)
print("ISSUE DIAGNOSIS:")
print("="*100)

# Check if non-detection rate increases with z
z_off_arr = np.array(z_offsets)
if len(z_offsets) > 3:
    # Correlation: does higher-z → more non-detections → mass bias?
    corr_z_nondet = np.corrcoef(z_off_arr[:, 0], z_off_arr[:, 1])[0, 1]
    print(f"\nCorrelation(z, non-detection-fraction): {corr_z_nondet:.3f}")
    
    if corr_z_nondet > 0.5:
        print("⚠️  HIGH-Z OBJECTS ARE MORE OFTEN NON-DETECTED")
        print("   → This could bias mass estimates!")
        
        # Check if background_noise limits are realistic
        print("\n📋 Background noise limits currently set to:")
        for fi, fname in enumerate(FILTER_SHORT[:3]):
            print(f"   {fname}: mag_err ≈ {mag_conversion(sx.limits[fi], convert_to='mag'):.2f} mag")
        print("   These are EXTREMELY deep (27-30 mag) — likely too optimistic!")
        print("\n💡 RECOMMENDATION:")
        print("   1. Recompute background_noise from COSMOS-Deep as proper 5σ depth limits")
        print("   2. Verify limits match COSMOSWeb's actual survey depths")
        print("   3. Re-test high-z mass estimates after correcting limits")
    elif corr_z_nondet > 0.0:
        print("⚠️  Slight tendency for higher-z to have more non-detections")
    else:
        print("✅ Non-detection rate does not increase with redshift")

print("\nNon-detection error assignment:")
print(f"  Current: mag=99.0, mag_err = background_noise limit")
print(f"  This effectively places non-detections at ~30 mag with huge error bars")
print(f"  MODEL SEES: Very bright non-detections with high uncertainty")
print(f"  → May systematically overpredict masses for faint/high-z population")

