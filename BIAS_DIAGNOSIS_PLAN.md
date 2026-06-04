# Complete Bias Diagnosis Plan

## The Problem (Summary)

- **Euclid bias**: +0.13 dex (z<0.5) → +1.37 dex (z>3) — **GROWS with redshift**
- **JWST bias**: +0.45 dex (flat across all z) — **CONSTANT with redshift**

This pattern tells us:
1. FSPS SED contributes a **flat +0.45 dex** floor
2. Euclid's shallow photometry adds **z-dependent amplification** (up to +0.92 dex at z>3)
3. JWST's deeper photometry **eliminates the z-dependent part**

## Three Root Cause Hypotheses

### Hypothesis A: FSPS M/L Calibration is Wrong
**Mechanism**: FSPS stellar mass-to-light ratios are systematically 0.45 dex too high.  
**Test**: Compare FSPS M/L to observational calibrations.  
**Fix**: Zero-point correction OR swap to different SED library (BC03, BPASS).

### Hypothesis B: Atlas sSFR Distribution is Biased  
**Mechanism**: Training atlas has fewer young/star-forming galaxies than real population.  
**Test**: Compare atlas sSFR distribution to COSMOS-Web/DESI main sequence.  
**Fix**: Retrain with importance weighting or rebalanced SFH prior.

### Hypothesis C: Age-Metallicity Degeneracy Not Resolved
**Mechanism**: No photometry (even JWST) can fully break age/Z degeneracy without spectroscopy.  
**Test**: See if different stellar population libraries give different bias.  
**Fix**: Use different library (BC03 vs FSPS vs BPASS).

---

## Server Diagnostic Workflow

### Step 1: Inspect FSPS Atlas Distribution (15 min)

```bash
python examples/inspect_fsps_ml.py \
    --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
    --outdir sbi-logs/fsps_inspection
```

**What it shows**:
- `fsps_distribution.png`: logM, sSFR, and (logM, logSFR) distributions in JWST atlas
- `fsps_vs_ms.png`: Atlas sSFR compared to Schreiber+2015 main sequence

**What to look for**:
- Is atlas median sSFR at logM=10 close to observed main sequence?
- At z=2, MS predicts log(sSFR) ≈ −8.5 for logM=10
- If atlas sSFR is −9.5 or lower, **Hypothesis B** is likely
- If atlas sSFR matches MS, then **Hypothesis A or C** is more likely

**Output**: Bias offset summary by mass bin

---

### Step 2: Generate Mock vs Real Diagnostic Grids (30 min)

```bash
python examples/diagnose_jwst_atlas.py \
    --outdir sbi-logs/diagnose_jwst_fullstack
```

**What it generates**:
- `jwst_mag_grid.png`: Median F277W, F444W magnitudes in (z, logM) cells
  - Real COSMOS-Web data
  - Expected from noise model
  
- `jwst_sigma_grid.png`: σ(mag) from noise model across parameter space

**What to look for**:
- Is the magnitude coverage uniform across (z, logM)?
- Are there "holes" (white cells) where data is sparse?
- Do σ values show expected trends (lower at bright, higher at faint)?
- Compare to your Euclid diagnostic — is JWST coverage better?

---

### Step 3: Compare Euclid Diagnostic (Already Have)

You already have:
```
sbi-logs/diagnose_v2.0_zfloor/mag_grid.png
sbi-logs/diagnose_v2.0_zfloor/sigma_grid.png
```

These show:
- Euclid real observed data vs mock predictions
- Delta (residuals) in (z, logM) cells
- **Key finding**: Large positive residuals at high-z (why Euclid bias grows)

---

### Step 4: Extract Schreiber+2015 Main Sequence (If Step 1 Unclear)

If atlas sSFR looks suspicious, quantify it:

```python
# Quick Python script
import numpy as np
import pickle

# Load atlas
with open("library/atlas_jwst_50000_Nparam_2.dbatlas", "rb") as f:
    atlas = pickle.load(f)

logM = atlas["theta"][:, 0]
logSFR = atlas["theta"][:, 1]
sSFR = logSFR - logM

# Schreiber+2015 at z=2
z = 2.0
logM_test = 10.0
logSFR_ms = np.log10(0.83) - 0.027 + 0.76 * np.log10(1+z) + \
           (np.log10(1+z) - 0.13) * (10**(logM_test - 10.5))
sSFR_ms = logSFR_ms - logM_test

print(f"Schreiber+15 MS at z={z}, logM={logM_test}: log(sSFR) = {sSFR_ms:.2f}")
print(f"Atlas median log(sSFR): {np.median(sSFR):.2f}")
print(f"Offset: {np.median(sSFR) - sSFR_ms:.2f} dex")
```

---

## Interpreting Results

### If Step 1 Shows sSFR Offset ≠ 0:
→ **Hypothesis B is likely true**  
→ Retrain atlas with corrected SFH prior  
→ Or use importance weighting to rebalance

**Correction formula** (if median atlas sSFR is Δ dex too low):
- Reweight each training galaxy by: `w = 10^(Δ * [logSFR_predicted - logSFR_median])`
- Retrain SBI with weighted likelihood

### If Step 1 Shows sSFR Matches MS:
→ **Hypothesis A or C is true**  
→ FSPS SED model itself is wrong  
→ Need to test library swap (BC03 vs FSPS)

**BC03 test**:
- Edit simulator.py to use BC03 instead of FSPS
- Regenerate 10k atlas subset
- Train quick model
- Run inference
- Compare bias to JWST +0.45 dex baseline

### If Step 2 Shows Uniform Coverage + Low σ:
→ **Good news**: Photometry model is reasonable  
→ Bias is not due to noise model being wrong  
→ Focus on SED library (Hypothesis A/C)

---

## Action Sequence (Recommended)

1. **Run Step 1** (inspect_fsps_ml.py) — 15 min
   - If sSFR is off: proceed to Hypothesis B fix
   - If sSFR matches: proceed to Hypothesis A/C tests

2. **Run Step 2** (diagnose_jwst_atlas.py) — 30 min
   - Sanity check that noise model is reasonable
   - Compare to Euclid diagnostic

3. **Based on Step 1 result**:
   - **Path A** (sSFR bias): Retrain with corrected prior → test on COSMOS-Web
   - **Path B** (FSPS library): Test BC03 swap → generate new atlas → retrain → test

4. **Measure improvement**: Re-run inference on COSMOS-Web master catalog
   - If bias drops below 0.3 dex → solution found
   - If not: iterate (may need multiple contributors)

---

## Expected Timings

| Task | Duration | Outputs |
|------|----------|---------|
| inspect_fsps_ml.py | 15 min | `fsps_distribution.png`, `fsps_vs_ms.png` |
| diagnose_jwst_atlas.py | 30 min | `jwst_mag_grid.png`, `jwst_sigma_grid.png` |
| Retrain atlas (if needed) | 2-3 hr | `atlas_new.dbatlas` |
| Train SBI model (if needed) | 2-3 hr | `model_new.pkl` |
| Inference test (500 gal) | 30 min | `mass_comparison.png`, bias statistic |

**Total**: 1-2 days if Path A needed; 1-2 weeks if library swap needed.

---

## Key Output to Monitor

**The single most important plot**: `fsps_vs_ms.png`

If atlas sSFR (red squares) tracks Schreiber+2015 MS (blue line):
→ **FSPS SED model is the problem** (not training data)

If atlas sSFR is systematically offset from MS:
→ **Training data is the problem** (fixable faster)
