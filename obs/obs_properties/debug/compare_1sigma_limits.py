"""Compare computed 1-sigma background limits with 5-sigma depth-based values."""

import numpy as np
from pathlib import Path

# ============================================================================
# Load computed 1-sigma limits from background_noise .npy file
# ============================================================================
rapid_test_limits = np.load('background_noise_rapid_test.npy')

# ============================================================================
# Compute 1-sigma from 5-sigma depth limits (from noise_5sigmadepth.py)
# ============================================================================
depth5_mag = {
    'Euclid_NISP.H': 24.25,
    'Euclid_NISP.J': 24.25,
    'Euclid_NISP.Y': 24.25,
    'Euclid_VIS.vis': 25.80,
    'Subaru_HSC.g': 26.80,
    'Subaru_HSC.z': 25.90,
    'CTIO_DECam.g': 26.46,
    'CTIO_DECam.r': 25.73,
    'CTIO_DECam.i': 25.54,
    'CTIO_DECam.z': 24.97,
}

# Filter order (from filters_to_use.dat)
order = [
    'Euclid_NISP.H',
    'Euclid_NISP.J',
    'Euclid_NISP.Y',
    'Euclid_VIS.vis',
    'Subaru_HSC.g',
    'Subaru_HSC.z',
    'CTIO_DECam.g',
    'CTIO_DECam.r',
    'CTIO_DECam.i',
    'CTIO_DECam.z',
]

def mag_to_flux_ujy(m):
    """Convert AB magnitude to flux in microJansky."""
    return 3631.0 * 1e6 * 10 ** (-0.4 * m)

# Compute 5-sigma flux for each filter
depth5_ujy = np.array([mag_to_flux_ujy(depth5_mag[k]) for k in order], dtype=float)

# Convert 5-sigma to 1-sigma by dividing by 5
depth1_ujy = depth5_ujy / 5.0

# ============================================================================
# Display Comparison
# ============================================================================
print("=" * 100)
print("BACKGROUND NOISE (1-SIGMA DETECTION LIMIT) COMPARISON")
print("=" * 100)
print()

print(f"{'Filter':<20s} | {'5σ Depth':<10s} | {'5σ Flux':<14s} | {'Computed 1σ':<14s} | {'From 5σ/5':<14s} | {'Difference':<12s} | {'Ratio':<8s}")
print("-" * 100)

for idx, (name, mag5) in enumerate([(k, depth5_mag[k]) for k in order]):
    computed = rapid_test_limits[idx]
    from_depth = depth1_ujy[idx]
    diff = computed - from_depth
    if from_depth > 0:
        ratio = computed / from_depth
    else:
        ratio = np.nan
    
    print(f"{name:<20s} | {mag5:<10.2f} | {depth5_ujy[idx]:<14.4f} | {computed:<14.6f} | {from_depth:<14.6f} | {diff:+.6f} | {ratio:.4f}")

print()
print("=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

diff_array = rapid_test_limits - depth1_ujy
ratio_array = rapid_test_limits / depth1_ujy

print(f"Mean difference (computed - from 5σ): {np.mean(diff_array):+.6f} μJy")
print(f"Std of difference:                    {np.std(diff_array):.6f} μJy")
print(f"Min difference:                       {np.min(diff_array):+.6f} μJy")
print(f"Max difference:                       {np.max(diff_array):+.6f} μJy")
print()
print(f"Mean ratio (computed / from 5σ):      {np.mean(ratio_array):.6f}")
print(f"Std of ratio:                         {np.std(ratio_array):.6f}")
print(f"Min ratio:                            {np.min(ratio_array):.6f}")
print(f"Max ratio:                            {np.max(ratio_array):.6f}")
print()
print(f"Filters within 10% of 5σ-based:       {np.sum(np.abs(ratio_array - 1.0) < 0.1)}/{len(order)}")
print(f"Filters within 20% of 5σ-based:       {np.sum(np.abs(ratio_array - 1.0) < 0.2)}/{len(order)}")
print()

# Show filters with largest differences
print("Filters with largest deviations from 5σ-based:")
print("-" * 100)
abs_ratio_diff = np.abs(ratio_array - 1.0)
for idx in np.argsort(-abs_ratio_diff)[:3]:
    name = order[idx]
    computed = rapid_test_limits[idx]
    from_depth = depth1_ujy[idx]
    ratio = computed / from_depth
    pct_diff = (ratio - 1.0) * 100
    print(f"  {name:<20s}  Ratio: {ratio:.4f}  ({pct_diff:+.1f}%)")

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print("""
The 1-sigma detection limits are computed from the COSMOS-Deep catalog data:
  • Computed limits: Empirical 1σ flux errors from the faint-end of real data
  • From 5σ depth: Theoretical estimates based on published 5σ detection limits

Differences can arise from:
  • Different survey conditions (COSMOS vs. EUCLID specifications)
  • Different aperture sizes (2fwhm/3fwhm pooling vs. published specs)
  • Empirical noise vs. theoretical predictions
  • Calibration differences between surveys

A ratio close to 1.0 indicates good agreement.
""")
print("=" * 100)
