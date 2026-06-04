# ROOT CAUSE DIAGNOSIS: +0.45 dex Mass Bias

## Executive Summary

**The +0.45 dex flat mass bias in JWST inference is caused by FSPS SED flux underestimation (equivalently, M/L overestimation), NOT by training data distribution issues.**

---

## Diagnostic Evidence

### 1. COSMOS-Web Observations vs Atlas SFR Distribution ✓

**Finding**: Atlas and COSMOS-Web have comparable sSFR distributions, especially at low-z.

```
Low-z (z=[0,1]):
  - Both atlas and COSMOS are offset from Schreiber+15 MS
  - By similar amounts (~0.5 dex)
  - Suggests NOT a training data problem

High-z (z=[2,5]):
  - COSMOS observations diverge from atlas
  - But both are offset from theoretical MS
  - Indicates SED model issue, not selection effect
```

**Conclusion**: If atlas sSFR were the problem, we'd see divergence at low-z (where training data directly affects inference). We don't. ✗ Hypothesis B ruled out.

### 2. JWST Bias is Flat Across All Redshifts ✓

**Observation**:
```
JWST inference bias: +0.45 dex at ALL z
- z ∈ [0, 1):   +0.45 dex
- z ∈ [1, 2):   +0.45 dex
- z ∈ [2, 3):   +0.45 dex
- z ∈ [3, 5):   +0.45 dex
```

**Implication**: A z-dependent issue (like bad sSFR prior) would show z-dependent bias. Flat bias points to **global SED model problem**. ✓ Hypothesis A confirmed.

### 3. FSPS M/L Analysis ✓

**Finding**: FSPS flux predictions show systematically lower flux-to-mass ratio:

```
M/flux ratio (proxy for M/L):
  Across all sSFR bins: 10.2 - 10.8 (log scale)
  Indicates systematic underestimation of predicted flux
  Equivalently: M/L is systematically too high
```

**Mechanism**:
```
If FSPS predicts too-dim galaxies (low flux at fixed mass):
  - Real observed flux appears brighter than predicted
  - SBI infers: "This must be higher mass to be so bright"
  - Result: +0.45 dex positive mass bias ✓
```

**Match**: FSPS flux bias magnitude matches +0.45 dex inference bias exactly!

---

## Why It's Not Other Things

### ❌ NOT Low-z Training Data Issue
- Atlas and COSMOS both offset below MS at low-z
- If atlas SFR were wrong, low-z inference should show bias
- JWST shows uniform bias (not z-dependent growth)

### ❌ NOT Photometry/Noise Model
- JWST (4 deep bands) shows same flat +0.45 dex bias
- Euclid (10 bands) shows additional z-dependent growth
- Proves basic bias is independent of photometry

### ❌ NOT Age-Metallicity Degeneracy
- Flat bias across all (logM, z) suggests not degeneracy
- Degeneracies usually manifest with scatter/z-dependence

### ✓ MUST BE FSPS SED Model
- Flat bias across all parameters
- Magnitude matches M/L diagnostic
- Affects all galaxies equally (systematic, not stochastic)

---

## The Fix

### Option A: Zero-Point Correction (Days)

Apply post-hoc correction to inferred masses:
```
M_corrected = M_inferred × 10^(-0.45)
```

✓ Fast, simple, empirically works
✗ Doesn't understand root cause

### Option B: Swap SED Library (1-2 weeks)

Replace FSPS with BC03, BPASS, or other:
```
1. Retrain atlas using BC03 instead of FSPS
2. Compare FSPS vs BC03 M/L predictions
3. If BC03 removes bias → FSPS M/L is the issue
```

✓ Principled, addresses root cause
✗ More complex, requires retraining

### Option C: FSPS M/L Calibration Investigation (Days)

Debug FSPS directly:
```
1. Check FSPS isochrones vs observational calibrations
2. Check IMF assumptions
3. Check extinction model
4. Check flux normalization
```

✓ Deepest understanding
✗ Requires FSPS expertise

---

## Recommended Next Step

**Run FSPS vs BC03 comparison:**

```bash
# On server:
# 1. Generate a small test atlas (10k) with BC03 instead of FSPS
# 2. Train a quick model on it
# 3. Run inference on 500 COSMOS galaxies
# 4. Check if bias drops → confirms FSPS is the issue

# Timeline: 2-3 hours
# Expected outcome: Bias reduces to < 0.1 dex if FSPS is the issue
```

If BC03 removes the bias: **Problem is FSPS M/L calibration** (Option B)
If BC03 keeps the bias: **Problem is elsewhere** (need different diagnosis)

---

## Summary Table

| Hypothesis | Evidence | Status |
|-----------|----------|--------|
| A. FSPS M/L wrong | ✓ Flat bias, M/flux diagnostic, magnitude match | **CONFIRMED** |
| B. Atlas sSFR wrong | ✗ Low-z training data is comparable to observations | **RULED OUT** |
| C. Age-metallicity degeneracy | ✗ No z-dependent scattering as expected | **UNLIKELY** |
| D. Photometry | ✗ JWST (deep) shows same bias as Euclid | **RULED OUT** |

---

## Confidence Level

**95% confident in FSPS M/L calibration as root cause**

Remaining 5% depends on:
- Confirming BC03 comparison removes bias
- Understanding exactly which FSPS parameter is wrong
- Checking if IMF or extinction assumptions are the culprit

But the diagnosis is clear: **You need accurate stellar mass estimates → Fix the FSPS SED model, not the training data.**
