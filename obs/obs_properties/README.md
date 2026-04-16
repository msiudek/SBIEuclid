# SBIPIX Noise Feature Generation

## Summary

Noise feature generation pipeline

## Quick Start

### Run 
```bash
cd /home/msiudek/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties
source ../../sbi_env/bin/activate
python learn_obs_noise_from_survey.py
```

### Files Generated
- `lam_eff.npy` (10 filter effective wavelengths)
- `percentiles_[aperture].npy` (magnitude bin boundaries: 6×10)
- `mean_sigma_[aperture].npy` (mean magnitude uncertainty: 10×7)
- `std_sigma_[aperture].npy` (std of magnitude uncertainty: 10×7)
- `sigma_samples_[aperture].npy` (raw samples: 10×7, ~78 MB)
- `background_noise_[aperture].npy` (detection limits: 10 values)

---

## How Each .npy File Is Generated

### 1. **lam_eff.npy** - Effective Wavelength Per Filter
- **Input:** Filter transmission curves
- **Calculation:** Weighted average of wavelength
- **Formula:** $\lambda_{eff} = \frac{\int \lambda T(\lambda) d\lambda}{\int T(\lambda) d\lambda}$
Where:
- $\lambda$ = wavelength from filter file (Angstrom)
- $T(\lambda)$ = transmission curve (0-1)
- Uses trapezoidal numerical integration
- **Output:** Effective wavelength in Angstrom for each filter

### File 2-5: Magnitude Uncertainty Statistics
**Pre-processing:**
1. Filter valid data: finite values, positive flux, SNR > 2
2. Convert flux to magnitude:
$$m = -2.5 \log_{10}\left(\frac{\text{flux\_ujy} \times 10^{-6}}{3631 \text{ Jy}}\right)$$
3. Compute magnitude uncertainty:
$$\delta m = \frac{2.5}{\ln(10)} \times \frac{\sigma_{\text{flux}}}{\text{flux}}$$

### 2. **percentiles_*.npy** - Magnitude Bin Boundaries
- **Input:** Galaxy magnitudes (531,811 sources)
- **Calculation:** For each filter, compute 5th, 15th, 30th, 50th, 70th, 90th percentiles
- **Output:** Magnitude threshold values that define bins

### 3. **mean_sigma_*.npy** - Mean Magnitude Uncertainty Per Bin
- **Input:** Magnitude uncertainties
- **Calculation:** For each filter and magnitude bin, compute mean uncertainty
- **Output:** Average magnitude uncertainty per filter and magnitude bin

### 4. **std_sigma_*.npy** - Std of Magnitude Uncertainty Per Bin
- **Input:** Magnitude uncertainties
- **Calculation:** For each filter and magnitude bin, compute standard deviation
- **Output:** Standard deviation of magnitude uncertainty per filter and magnitude bin

### 5. **sigma_samples_*.npy** - Raw Magnitude Error Samples
- **Input:** Magnitude uncertainties
- **Calculation:** Store raw error arrays per filter/bin
- **Output:** Raw empirical magnitude error samples for later analysis
- **Note:** Large file (~78 MB) because it stores all raw samples

### 6. **background_noise_*.npy** - 1-Sigma Detection Limits
- **Input:** Flux values and their uncertainties (finite, positive, SNR > 2)
- **Calculation:** For each filter, find faint-end sources (20th percentile) and compute median flux error
- **Output:** 1-sigma flux detection limit in microJansky per filter

## One-Minute Overview

1. **Load:** 10 filter transmission curves + 949K galaxy photometry
2. **Process:** 
   - Compute effective wavelength per filter
   - Convert flux to magnitude
   - Bin galaxies by brightness
   - Compute magnitude uncertainty stats per bin
3. **Output:** 11 .npy files with calibrated noise model
4. **Time:** ~25 seconds total


### Input Files (must exist)
- `COSMOS_DEEP.fits` (446 MB)
  - FITS catalog with flux measurements
  - Column: `patch_id_list` for patching
  - Columns: `flux_<stem>_<aperture>_aper` for photometry
  - Columns: `fluxerr_<stem>_<aperture>_aper` for uncertainties

- `filters_to_use.dat` (text file)
  - List of filter file paths (relative to FILTER_DIR)
  - Format: one path per line, blank lines and `#` comments ignored

- `FILTERS_*/*.dat` (10 files)
  - Filter transmission curves
  - Columns: wavelength (Angstrom), transmission (0-1)
  - Used to compute effective wavelength



### Notes
- We are using empirical 1-σ limits (from COSMOS-Deep) instead of 5-σ depth limits as emiprical ones are 50% lower than theoretical ones suggesting that COSMOS-Deep has better sensitivity (less noise) than expected from published 1-σ depth limits --> COSMOS-Deep appears deeper/better than that theoretical reference for most bands. This could be a result of focusing on a calibration run. The comparison script is given in compare_1sigma_limits.py

- The output is generated for two apertures, while 3fwhm is noisier than 2fwhm as expected from larger aperture.

- we are limiting the dataset to calibration subset (PatchID 98)
https://euclid.roe.ac.uk/projects/sgs-ops-procedures/wiki/DR1_Reprocessing#DEEP-fields
- the data were taken from https://easidr.esac.esa.int/sas/ with a query:
SELECT 
    mer.object_id,
    mer.right_ascension, mer.declination,
    mer.flux_detection_total,
    mer.spurious_flag,
    mer.det_quality_flag,
    mer.mumax_minus_mag,
    flux_u_ext_megacam_2fwhm_aper,
    flux_g_ext_hsc_2fwhm_aper,
    flux_r_ext_megacam_2fwhm_aper,
    flux_i_ext_panstarrs_2fwhm_aper,
    flux_z_ext_hsc_2fwhm_aper,

    flux_g_ext_decam_2fwhm_aper,
    flux_r_ext_decam_2fwhm_aper,
    flux_i_ext_decam_2fwhm_aper,
    flux_z_ext_decam_2fwhm_aper,

    flux_vis_2fwhm_aper,
    flux_y_2fwhm_aper,
    flux_j_2fwhm_aper,
    flux_h_2fwhm_aper,
    fluxerr_u_ext_megacam_2fwhm_aper,
    fluxerr_g_ext_hsc_2fwhm_aper,
    fluxerr_r_ext_megacam_2fwhm_aper,
    fluxerr_i_ext_panstarrs_2fwhm_aper,
    fluxerr_z_ext_hsc_2fwhm_aper,

    fluxerr_g_ext_decam_2fwhm_aper,
    fluxerr_r_ext_decam_2fwhm_aper,
    fluxerr_i_ext_decam_2fwhm_aper,
    fluxerr_z_ext_decam_2fwhm_aper,

    fluxerr_vis_2fwhm_aper,
    fluxerr_y_2fwhm_aper,
    fluxerr_j_2fwhm_aper,
    fluxerr_h_2fwhm_aper,
    flux_u_ext_megacam_3fwhm_aper,
    flux_g_ext_hsc_3fwhm_aper,
    flux_r_ext_megacam_3fwhm_aper,
    flux_i_ext_panstarrs_3fwhm_aper,
    flux_z_ext_hsc_3fwhm_aper,
    flux_g_ext_decam_3fwhm_aper,
    flux_r_ext_decam_3fwhm_aper,
    flux_i_ext_decam_3fwhm_aper,
    flux_z_ext_decam_3fwhm_aper,
    flux_vis_3fwhm_aper,
    flux_y_3fwhm_aper,
    flux_j_3fwhm_aper,
    flux_h_3fwhm_aper,
    fluxerr_u_ext_megacam_3fwhm_aper,
    fluxerr_g_ext_hsc_3fwhm_aper,
    fluxerr_r_ext_megacam_3fwhm_aper,
    fluxerr_i_ext_panstarrs_3fwhm_aper,
    fluxerr_z_ext_hsc_3fwhm_aper,

    fluxerr_g_ext_decam_3fwhm_aper,
    fluxerr_r_ext_decam_3fwhm_aper,
    fluxerr_i_ext_decam_3fwhm_aper,
    fluxerr_z_ext_decam_3fwhm_aper,

    fluxerr_vis_3fwhm_aper,
    fluxerr_y_3fwhm_aper,
    fluxerr_j_3fwhm_aper,
    fluxerr_h_3fwhm_aper,
    
    flux_u_ext_megacam_templfit,
    flux_g_ext_hsc_templfit,
    flux_r_ext_megacam_templfit,
    flux_i_ext_panstarrs_templfit,
    flux_z_ext_hsc_templfit,

    flux_g_ext_decam_templfit,
    flux_r_ext_decam_templfit,
    flux_i_ext_decam_templfit,
    flux_z_ext_decam_templfit,

    flux_vis_templfit,
    flux_y_templfit,
    flux_j_templfit,
    flux_h_templfit,
    fluxerr_u_ext_megacam_templfit,
    fluxerr_g_ext_hsc_templfit,
    fluxerr_r_ext_megacam_templfit,
    fluxerr_i_ext_panstarrs_templfit,
    fluxerr_z_ext_hsc_templfit,

    fluxerr_g_ext_decam_templfit,
    fluxerr_r_ext_decam_templfit,
    fluxerr_i_ext_decam_templfit,
    fluxerr_z_ext_decam_templfit,

    fluxerr_vis_templfit,
    fluxerr_y_templfit,
    fluxerr_j_templfit,
    fluxerr_h_templfit,
    
    flux_u_ext_megacam_sersic,
    flux_g_ext_hsc_sersic,
    flux_r_ext_megacam_sersic,
    flux_i_ext_panstarrs_sersic,
    flux_z_ext_hsc_sersic,
    flux_g_ext_decam_sersic,
    flux_r_ext_decam_sersic,
    flux_i_ext_decam_sersic,
    flux_z_ext_decam_sersic,
    flux_vis_sersic,
    flux_y_sersic,
    flux_j_sersic,
    flux_h_sersic,
    fluxerr_u_ext_megacam_sersic,
    fluxerr_g_ext_hsc_sersic,
    fluxerr_r_ext_megacam_sersic,
    fluxerr_i_ext_panstarrs_sersic,
    fluxerr_z_ext_hsc_sersic,

    fluxerr_g_ext_decam_sersic,
    fluxerr_r_ext_decam_sersic,
    fluxerr_i_ext_decam_sersic,
    fluxerr_z_ext_decam_sersic,

    fluxerr_vis_sersic,
    fluxerr_y_sersic,
    fluxerr_j_sersic,
    fluxerr_h_sersic,

    flag_j, flag_h, flag_y, flag_vis,
    patch_id_list

FROM catalogue.mer_catalogue_deep AS mer

WHERE 
    mer.right_ascension BETWEEN 149.66412142262124 AND 150.5769686752853
	AND mer.declination BETWEEN 1.7293767764387382 AND 2.6872526337031895

    AND mer.flux_detection_total > 0
    AND mer.spurious_flag = 0 

    AND mer.det_quality_flag  = 0 
    AND mer.mumax_minus_mag > -2.6

