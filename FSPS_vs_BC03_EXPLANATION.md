# FSPS vs BC03: Understanding the Differences

## What Are SSP Models?

**Simple Stellar Population (SSP)** = single generation of stars formed at same time with same metallicity.

For each SSP at age t and metallicity Z, we compute:
- Spectrum λ(t, Z)
- Luminosity L(t, Z)  
- Stellar mass M*(t, Z)
- M/L ratio = M*(t, Z) / L(t, Z)

A galaxy is modeled as a sum of SSPs with different ages and metallicities (star formation history).

---

## FSPS: What It Uses

**FSPS** = Flexible Stellar Population Synthesis (Conroy+ 2009)

**Default stellar library**:
- **Isochrones**: Padova 2007/2008 (updated evolutionary tracks)
- **Stellar atmospheres**: Castelli & Kurucz models + others
- **TP-AGB stars**: Detailed treatment (matters for intermediate-age populations)
- **IMF**: Flexible (Chabrier, Kroupa, Salpeter, etc.)

**Key characteristic**: FSPS is modern, flexible, uses latest isochrones.

---

## BC03: What It Is

**BC03** = Bruzual & Charlot 2003 spectral synthesis models

**Stellar library**:
- **Isochrones**: Padova 1994 (older!)
- **Stellar atmospheres**: Gunn & Stryker (1989) library (older!)
- **TP-AGB stars**: Simple treatment (color-based approximation)
- **IMF**: Salpeter or Chabrier

**Key characteristic**: BC03 is older but well-established, more conservative.

---

## Why M/L Differs Between Them

The ~10 dex offset in your diagnostic plot comes from **how stellar mass is defined**:

### FSPS Definition
```
M*(t, Z) = Initial Mass Function integrated over stars
         = "Mass of stars ever formed" (includes dead remnants)
         = Stellar mass formed (cumulative)
```

### BC03 Definition  
```
M*(t, Z) = Mass of stars currently alive
         = "Surviving stellar mass" (excludes stellar remnants)
         = Stellar mass surviving
```

**Example: 13 Gyr old population**
- FSPS: counts white dwarfs + neutron stars in mass
- BC03: does not count white dwarfs/neutron stars

This can cause **0.1-0.3 dex difference** in M/L ratios!

---

## Why This Matters for Your Bias

Your +0.45 dex bias could come from:

1. **FSPS using wrong isochrones** (Padova 2007 may be biased)
2. **FSPS using formed vs surviving mass inconsistently**
3. **FSPS stellar library outdated** (TP-AGB treatment)
4. **Mismatch between FSPS definition and LePhare/observations**

If LePhare uses BC03-style "surviving mass" but FSPS uses "formed mass", systematic offset is expected!

---

## How to Fix This: Two Approaches

### Approach 1: Change FSPS Configuration (Easy)

FSPS has parameters to change stellar libraries:

```python
# In simulator.py or sbi_setup:
sps = fsps.StellarPopulation()
sps.isochrone_type = 'padova_2007'    # Current default
# Try alternatives:
sps.isochrone_type = 'padova_1994'    # Older, might match observations better
sps.include_nebular = False            # Toggle nebular emission
sps.add_stellar_remnants = False       # Exclude white dwarfs/neutron stars!
```

**This is the fastest fix!** Just exclude stellar remnants and see if bias drops.

### Approach 2: Use BC03 Directly (Medium)

BC03 is available standalone:

```bash
# Download from: http://www.bruzual.org/bc03/
# Use with fsps:
sps = fsps.StellarPopulation()
sps.sspnelyc = 'bc03/sed_files/'  # Point to BC03 files
```

Or use a different code entirely (GALAXEV, STARBURST99).

### Approach 3: Empirical Correction (Fast)

Apply post-hoc zero-point fix to inferred masses:

```python
# After SBI inference:
M_corrected = M_inferred × 10^(-0.45)
```

✓ Fast, works immediately
✗ Doesn't fix the underlying issue

---

## Stellar Mass Definitions: YOUR KEY INSIGHT

You're absolutely right that there are TWO mass definitions:

| Definition | Includes | Example Age |
|-----------|----------|------------|
| **Formed mass** | All stars ever born, including dead remnants (WD, NS) | 13 Gyr pop: 1.5 × 10^11 M☉ |
| **Surviving mass** | Only living stars | 13 Gyr pop: 1.3 × 10^11 M☉ |

**FSPS default**: Formed mass (includes all)
**BC03/Observations**: Might be surviving mass (excludes dead stars)

**The Fix**: 
```python
# In FSPS setup, disable stellar remnants:
sps.add_stellar_remnants = False
# Now FSPS returns surviving mass, matching observations
```

This could account for **0.1-0.3 dex** of your bias!

---

## Recommended Test Plan

**Day 1: Quick FSPS Configuration Test**

```python
# Test 1: Exclude stellar remnants
sps.add_stellar_remnants = False
# Generate quick 5k atlas
# Train fast model
# Run inference
# Check bias → if drops to 0.2 dex, you found it!

# Test 2: Try Padova 1994 isochrones  
sps.isochrone_type = 'padova_1994'
# Same process
# Check if bias changes

# Test 3: Disable TP-AGB
sps.add_agb_dust = False
# Check impact
```

**Expected results**:
- If `add_stellar_remnants=False` drops bias to 0.2 dex → **This is the issue!**
- If isochrone change helps → **Padova 2007 may be biased**
- If TP-AGB matters → Intermediate-age stars are the culprit

---

## Adding Second Mass Definition to Inference

Yes, absolutely! Modify `inference_cosmosweb_jwst.py`:

```python
# In load_observations():
logM_formed = ref_hdu["mass_med"]        # "Formed mass" definition
logM_surviving = ref_hdu["mass_living"]  # "Surviving mass" definition (if available)

# Or compute: logM_surviving ≈ logM_formed - correction_age(z)
# Older galaxies have bigger correction
```

**In output plots**:
```python
# Plot both:
ax.scatter(logM_formed, logSFR_sbi, alpha=0.3, label="vs Formed Mass")
ax.scatter(logM_surviving, logSFR_sbi, alpha=0.3, label="vs Surviving Mass")
# See which one has lower bias!
```

This is a **critical diagnostic**: if bias drops when you switch to surviving mass, you've found the root cause!

---

## Summary: Your Path Forward

1. **Check FSPS config first** (1 hour):
   - Try `add_stellar_remnants = False`
   - Try different `isochrone_type`
   - Quick 5k atlas test

2. **If bias doesn't drop enough**: Test BC03 (1-2 days)

3. **Add mass definition comparison** to inference plots (1 hour)
   - Plot both formed and surviving mass
   - See which matches better

4. **Most likely outcome**: Stellar remnants definition is the issue, fixed by 1 line of code change!

This is more elegant than expected — the +0.45 dex bias might be **simply a definition mismatch** between what FSPS computes and what the observations measure.
