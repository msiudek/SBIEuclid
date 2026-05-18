# Testing Sigma Mismatch Impact on Faint Galaxies

## Quick Test: Run inference on faint NISP-Y mag 27-28 galaxies

```bash
# Test 1: Run inference on a small sample of very faint galaxies
python examples/inference_cosmosweb.py \
    --n-gal 50 \
    --mag-band NISP-Y \
    --mag-min 27.0 \
    --mag-max 28.0 \
    --model-name model_euclid_v9.0.pkl \
    --phot-type templfit \
    --sample-with rejection \
    --observation-space flux \
    --n-samples 200 \
    --device cuda \
    --n-bands-min 7 \
    --snr-min 3 \
    --outdir sbi-logs/test_faint_nispy_27_28

# Test 2: Compare with bright galaxies (mag 22-24)
python examples/inference_cosmosweb.py \
    --n-gal 50 \
    --mag-band NISP-Y \
    --mag-min 22.0 \
    --mag-max 24.0 \
    --model-name model_euclid_v9.0.pkl \
    --phot-type templfit \
    --sample-with rejection \
    --observation-space flux \
    --n-samples 200 \
    --device cuda \
    --n-bands-min 7 \
    --snr-min 3 \
    --outdir sbi-logs/test_bright_nispy_22_24
```

## Comprehensive Test: Run full diagnostic suite

```bash
python test_faint_galaxies.py --model-name model_euclid_v6.3_20k.pkl
```

This will:
1. Run inference on multiple magnitude ranges (bright, intermediate, faint, very faint)
2. Check for degenerate posteriors (width < 0.05 dex)
3. Compute mass bias and correlation with redshift
4. Generate comparison plots
5. Recommend whether background_noise calibration needs updating

## What to Look For

**Sanity Checks for Faint Galaxy Inference:**
1. **Posteriors not degenerate**: σ(logM*) > 0.1 dex (or results are garbage)
2. **Reasonable bias**: |bias| < 0.5 dex (anything larger suggests model issues)
3. **Bias not strongly z-dependent**: corr(bias, z) < 0.3 (indicates stable inference)
4. **Detection fractions make sense**: Should be similar to training regime or indicate mismatch

**If Very Faint Galaxies Show Large Bias or Degenerate Posteriors:**
- Problem 1: `background_noise_north_templfit.npy` doesn't match actual templfit depths
  - Solution: Run `validate_noise_model.py` and examine `sigma_vs_mag_*.png` plots
  - Check if templfit limits are shallower than assumed
- Problem 2: Noise model σ(mag) is poorly calibrated at high magnitudes
  - Solution: Re-fit `mean_sigma_north_templfit.npy`, `std_sigma_north_templfit.npy`
- Problem 3: Training data insufficient for faint regime
  - Solution: Use larger atlas (100k instead of 20k)

## Available Magnitude Bands

Use `--mag-band` with any of these filter names:
- NISP-H, NISP-J, NISP-Y (Euclid NIR)
- VIS (Euclid visible)
- HSC-g, HSC-z (Subaru)
- DECam-g, DECam-r, DECam-i, DECam-z (DECam)
