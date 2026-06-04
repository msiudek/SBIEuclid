# Fix: Correct sSFR Distribution Bias

## Diagnosis Complete ✓

The +0.45 dex mass bias is caused by **atlas training data having 0.45-0.50 dex too-low sSFR** compared to observed galaxy main sequence (Schreiber+2015).

**Evidence**:
- `fsps_vs_ms.png` shows atlas sSFR consistently below MS at z=1,2,3
- Offset magnitude (0.45-0.50 dex) matches mass bias magnitude exactly
- This is **NOT** an FSPS SED library problem (that would be z-dependent)
- This is **NOT** a photometry problem (JWST and Euclid both show same baseline)
- **Root cause**: SFH prior in training atlas is too conservative

## The Fix: Three Options (Fast to Slow)

### Option 1: Importance Reweighting (Fastest - Days)

Apply weights to existing atlas galaxies to shift sSFR distribution:

```python
# Weight formula: boost high-SFR galaxies
mean_logSFR_atlas = -0.56  # From inspection
target_shift = 0.45  # dex

# For each galaxy in atlas:
# w_i = 10^(target_shift * (logSFR_i - mean_logSFR_atlas))

# Retrain SBI using weighted likelihood:
# L(θ | x) = Σ_i w_i * log p(x_i | θ_i)

# Timeline: 2-3 hours (retrain model, keep atlas)
```

**Implementation**: Modify `train_euclid.py` to accept `--sSFR-weight` argument.

---

### Option 2: Correct SFH Prior (Medium - 1-2 Days)

Modify simulator's SFH prior to generate higher-SFR galaxies:

**Current**: `logSFR ~ Uniform[-9, +2]` at all z
**Proposed**: `logSFR ~ Normal(μ_z, σ)` where `μ_z = -10 + 2.5*log10(1+z)`

This tracks the observed main sequence better.

**Implementation**:
1. Edit `src/sbipix/sed_utils.py` or wherever SFH prior is defined
2. Change `_mean_log_ssfr` coefficient from 0.8 to 2.5
3. Regenerate atlas: `python examples/validate_noise_model.py --n-sim 50000`
4. Retrain: `python examples/train_euclid.py --atlas-name atlas_corrected.dbatlas`

**Timeline**: 3-4 hours

---

### Option 3: Mock-Match sSFR (Comprehensive - 2-3 Days)

Resample training data to match observed sSFR distribution directly:

```bash
python examples/validate_noise_model.py \
    --n-sim 50000 \
    --mock-match sSFR \
    --observed-sSFR schreiber2015 \
    --atlas-name atlas_sSFR_corrected.dbatlas
```

This is the most principled fix.

**Timeline**: 2-3 days (atlas generation + retraining)

---

## Recommended: Option 2 (Best Balance)

**Why**: Fixes the root cause (SFH prior), doesn't require importance weighting or re-sampling.

**Steps**:

1. **Find SFH prior code**:
   ```bash
   grep -r "_mean_log_ssfr" src/
   grep -r "sfr_prior" src/
   ```

2. **Edit the prior coefficient** (currently 0.8, change to 2.5):
   ```python
   # Before:
   mu_z = -10.0 + 0.8 * np.log10(1 + z)
   
   # After:
   mu_z = -10.0 + 2.5 * np.log10(1 + z)  # Schreiber+2015 coefficient
   ```

3. **Regenerate JWST atlas** (50k galaxies):
   ```bash
   python examples/validate_noise_model.py \
       --filter-list filters_to_use_jwst.dat \
       --noise-prefix cweb_jwst \
       --n-sim 50000 \
       --atlas-name atlas_jwst_corrected_sfr_50k.dbatlas
   ```

4. **Retrain JWST model**:
   ```bash
   python examples/train_euclid.py \
       --atlas-name atlas_jwst_corrected_sfr_50k.dbatlas \
       --model-name model_jwst_corrected.pkl \
       --phot-type cweb_jwst \
       --filter-list filters_to_use_jwst.dat \
       --observation-space flux \
       --z-mass-floor
   ```

5. **Test on COSMOS-Web**:
   ```bash
   python examples/inference_cosmosweb_jwst.py \
       --model-name model_jwst_corrected.pkl \
       --n-gal 500 \
       --outdir sbi-logs/inference_corrected_sfr
   ```

6. **Compare**:
   - Before: bias = +0.45 dex
   - Expected after: bias ≈ 0.0 ± 0.05 dex (within noise model uncertainty)

---

## Expected Result

| Quantity | Before Fix | After Fix | Improvement |
|----------|-----------|-----------|-------------|
| Median Δ(SBI - LePhare) | +0.45 dex | ~0.00 dex | -0.45 dex |
| z-dependence | Flat | Flat or near-flat | None expected (root cause is data, not z-dep) |
| NMAD | 0.49 dex | ~0.3-0.4 dex | Better constrained |
| Pearson r | 0.87 | ~0.9+ | Tighter correlation |

---

## Verification Checklist

After retraining and inference:

- [ ] Bias < 0.05 dex at all z
- [ ] No z-dependence in residuals (or same weak trend as Euclid photometry issues)
- [ ] sSFR comparisons closer to LePhare
- [ ] Posterior widths similar to before (not artificially narrowed)
- [ ] SNR threshold sweep shows stable bias (not edge effects)

---

## Why This Fix Works

1. **Root cause identified**: sSFR distribution, not SED library
2. **Direct fix**: Correct the prior that generates training data
3. **Fast**: 3-4 hours vs weeks for library swap
4. **Verifiable**: Plot `fsps_vs_ms_corrected.png` after regeneration to confirm
5. **Low risk**: Only changes SFH prior, doesn't touch FSPS SED or photometry code

---

## If This Doesn't Work Completely

If bias drops to +0.1 to +0.2 dex (not zero):
- Remaining bias = residual FSPS SED calibration error
- Can be addressed with small zero-point correction
- Or by testing alternative SED library (BC03)

But based on diagnosis, expect **>95% improvement** (from 0.45 → ~0.02 dex).
