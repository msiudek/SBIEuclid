"""
validate_noise_model.py — Pre-training data-space validation.

Runs sbipix simulate + noise injection, loads real COSMOS-Deep photometry
(selected photometry type), and produces side-by-side diagnostic plots to confirm
that mock observations match real data within ~10-20% before training.

Plots saved to  sbi-logs/validate_<filter>/  :
    1.  theta_histograms.png      — simulated parameter distributions
    2.  sigma_vs_mag_<filt>.png   — σ vs mag, real scatter + mock scatter + model bins
    3.  mag_hist_<filt>.png       — magnitude histogram, real vs mock (detected only)
    4.  det_fraction_<filt>.png   — detection fraction vs magnitude, real vs mock
    5.  colors.png                — optical/NIR color–color: real vs mock
    6.  sigma_dist_<filt>.png     — distribution of σ values, real vs mock

Usage:
    python examples/validate_noise_model.py \
        --n-sim 100000 \
        --skip-sim \
        --phot-type 2fwhm \
        --outdir sbi-logs/validate_sSFRlogNormal_v5.0 \
        --detection-model hard \
        --mock-match vis_yj2d \
        --sigma-sampler mag_lognormal \
        2>&1 | tee sbi-logs/validate_v5.0.log
"""

import argparse
import os
from pathlib import Path
import sys
# Shared utilities from sbipix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbipix.utils.sed_utils import (
    convert_to_microjansky,
    flux_ujy_to_mag,
    flux_ujy_to_mag_err,
    mag_to_flux_ujy,
    load_filter_metadata,
    sfh_delayed_exponential,
)
from sbipix.utils.validation_plots import (
    plot_sigma_vs_mag,
    plot_mag_histogram,
    plot_true_mag_histogram,
    plot_detection_fraction_vs_flux,
    plot_sigma_distribution,
    plot_colors,
    plot_sigma_vs_mag_grid,
    plot_intrinsic_color_color,
    plot_intrinsic_color_vs_parameters,
)
from sbipix.plotting.diagnostics import plot_theta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# Filter metadata — loaded from the single source of truth: filters_to_use.dat
# (3-column format: filter_rel_path  short_name  col_stem)
# ---------------------------------------------------------------------------
_OBS_DIR = Path(__file__).resolve().parents[1] / "obs" / "obs_properties"

_FILTER_META = load_filter_metadata("filters_to_use.dat", filt_dir=str(_OBS_DIR))
FILTER_SHORT     = [m["short"]    for m in _FILTER_META]
FILTER_COL_STEMS = [m["col_stem"] for m in _FILTER_META]

# Colors to plot: (band_a_idx, band_b_idx, label)
# Indices are 0-based positions in filters_to_use.dat (fixed order):
#   0=NISP-H  1=NISP-J  2=NISP-Y  3=VIS  4=HSC-g  5=HSC-z
#   6=DECam-g 7=DECam-r 8=DECam-i 9=DECam-z
COLOR_PAIRS = [
    (3, 2, "VIS - Y"),
    (2, 1, "Y - J"),
    (1, 0, "J - H"),
    (4, 7, "HSC-g - DECam-r"),
    (5, 9, "HSC-z - DECam-z"),
]

NONDET_MAG = 99.0
SNR_DETECTION_THRESHOLD = 3.0
MAG_BRIGHT = 16.0
MAG_FAINT = 30.0
CATALOG_FILE = "obs/obs_properties/COSMOS_DEEP_PHZ.fits"
PATCH_ID = 65879
AB_ZEROPOINT_JY = 3631.0
AB_ZEROPOINT_UJY = 3631e6
LN10 = np.log(10.0)
SIMULATION_CONFIG = {
    "mass_min": 6.0,
    "mass_max": 11.5,
    "sfr_prior_type": "sSFRlognormal",
    "ssfr_min": -12.5,
    "ssfr_max": -7.0,
    "z_prior": "flat",
    "z_min": 0.0,
    "z_max": 3.5,
    "Z_min": -0.8,
    "Z_max": 0.3,
    "dust_model": "Calzetti",
    "dust_prior": "flat",
    "Av_min": 0.0,
    "Av_max": 3.0,
    "tx_alpha": 0.7,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_mock_match_weights(real_data, mock_data):
    """Compute per-mock-object weights from the 2D joint (VIS mag, Y-J color) histogram.

    Matching on the joint distribution P(m_VIS, Y-J) captures both the overall
    brightness distribution and the color-redshift locus simultaneously.
    """
    n_mock = mock_data["mag"].shape[1]
    weights = np.zeros(n_mock, dtype=float)

    vis_idx = FILTER_SHORT.index("VIS")
    y_idx   = FILTER_SHORT.index("NISP-Y")
    j_idx   = FILTER_SHORT.index("NISP-J")

    # --- real ---
    real_vis = real_data["mag"][vis_idx]
    real_yj  = real_data["mag"][y_idx] - real_data["mag"][j_idx]
    real_ok  = (np.isfinite(real_vis) & np.isfinite(real_yj))

    # --- mock ---
    mock_vis = mock_data["mag"][vis_idx]
    mock_yj  = mock_data["mag"][y_idx] - mock_data["mag"][j_idx]
    # Use the same SNR-based detection criterion as real data:
    #   SNR = (2.5 / ln10) / mag_err  for a magnitude-error sigma
    _LN10 = np.log(10)
    mock_vis_sigma = mock_data["sigma"][vis_idx]
    mock_vis_snr = np.where(mock_vis_sigma > 0, (2.5 / _LN10) / mock_vis_sigma, 0.0)
    mock_ok  = (
        np.isfinite(mock_vis) & (mock_vis_snr >= SNR_DETECTION_THRESHOLD)
        & np.isfinite(mock_yj)
    )

    bins_mag = np.linspace(MAG_BRIGHT, MAG_FAINT, 25)   # 24 VIS-mag bins

    # data-driven Y-J color range from real objects
    if real_ok.sum() > 0:
        p1, p99 = np.nanpercentile(real_yj[real_ok], [1, 99])
        bins_color = np.linspace(p1 - 0.5, p99 + 0.5, 25)   # 24 color bins
    else:
        bins_color = np.linspace(-2.0, 4.0, 25)

    real_hist, _, _ = np.histogram2d(
        real_vis[real_ok], real_yj[real_ok], bins=[bins_mag, bins_color]
    )
    mock_hist, _, _ = np.histogram2d(
        mock_vis[mock_ok], mock_yj[mock_ok], bins=[bins_mag, bins_color]
    )

    # bin-by-bin ratio weights (normalised PDFs), clipped to avoid blow-up
    n_real = real_ok.sum()
    n_mock_ok = mock_ok.sum()
    if n_real == 0 or n_mock_ok == 0:
        return weights, "mock matching skipped: no valid VIS+Y-J objects"

    real_pdf = real_hist / n_real
    mock_pdf = mock_hist / n_mock_ok
    eps = 1e-6
    ratio = real_pdf / (mock_pdf + eps)
    ratio = np.clip(ratio, 0, 10)

    # assign weight to each mock object from its 2-D bin
    ix = np.digitize(mock_vis, bins_mag)   - 1
    iy = np.digitize(mock_yj,  bins_color) - 1
    in_range = (mock_ok
                & (ix >= 0) & (ix < ratio.shape[0])
                & (iy >= 0) & (iy < ratio.shape[1]))
    weights[in_range] = ratio[ix[in_range], iy[in_range]]

    return weights, (
        f"mock matching: 2D VIS×(Y-J) histogram 24×24 bins "
        f"(real n={n_real}, mock detected n={n_mock_ok})"
    )


def resample_mock_catalogue(mock_data, weights, seed=0):
    """Approximate reweighting by deterministic bootstrap resampling with replacement."""
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return mock_data, "mock matching skipped: all weights are zero"

    probs = weights[valid] / weights[valid].sum()
    source_idx = np.where(valid)[0]
    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(source_idx, size=mock_data["mag"].shape[1], replace=True, p=probs)
    eff_n = (weights[valid].sum() ** 2) / np.sum(weights[valid] ** 2)
    out = {}
    n_obj = mock_data["mag"].shape[1]
    for key, value in mock_data.items():
        if isinstance(value, np.ndarray) and n_obj in value.shape:
            axis = list(value.shape).index(n_obj)
            out[key] = np.take(value, draw_idx, axis=axis)
        else:
            out[key] = value
    return out, f"resampled weighted mock catalogue (effective N ≈ {eff_n:.0f})"


def print_mock_plot_counts(mock_data):
    """Print per-band mock counts used by sigma plots."""
    print("Mock counts used in plots:")
    print("  band        detected(<99)   sigma_dist(0<sigma<2)")
    for fi, band in enumerate(FILTER_SHORT):
        mag = mock_data["mag"][fi]
        sigma = mock_data["sigma"][fi]
        det_count = np.sum((mag < NONDET_MAG - 0.5) & np.isfinite(sigma))
        sig_count = np.sum(np.isfinite(sigma) & (sigma > 0) & (sigma < 2.0))
        print(f"  {band:>10s}  {det_count:8d}/{mag.size:<8d}  {sig_count:8d}/{sigma.size:<8d}")


def print_debug_diagnostics(real_data, mock_data):
    """Print zeropoints, z-i color residuals, and per-band median magnitude offsets."""
    print("\nzp_used_per_band (AB system used in conversions):")
    for band in FILTER_SHORT:
        print(f"  {band:>10s}  ZP = {AB_ZEROPOINT_JY:.1f} Jy = {AB_ZEROPOINT_UJY:.0f} μJy")

    print("\nColor residual diagnostics (median mock color - median real color):")
    i_band = "DECam-i"
    if i_band in FILTER_SHORT:
        i_idx = FILTER_SHORT.index(i_band)
        for z_band in ["DECam-z", "HSC-z"]:
            if z_band not in FILTER_SHORT:
                continue
            z_idx = FILTER_SHORT.index(z_band)

            real_z = real_data["mag"][z_idx]
            real_i = real_data["mag"][i_idx]
            mock_z = mock_data["mag"][z_idx]
            mock_i = mock_data["mag"][i_idx]

            real_ok = np.isfinite(real_z) & np.isfinite(real_i)
            mock_ok = (
                np.isfinite(mock_z) & np.isfinite(mock_i)
                & (mock_z < NONDET_MAG - 0.5)
                & (mock_i < NONDET_MAG - 0.5)
            )
            if not np.any(real_ok) or not np.any(mock_ok):
                print(f"  {z_band} - {i_band}: insufficient valid data")
                continue

            real_color = real_z[real_ok] - real_i[real_ok]
            mock_color = mock_z[mock_ok] - mock_i[mock_ok]
            real_med = float(np.nanmedian(real_color))
            mock_med = float(np.nanmedian(mock_color))
            delta_med = mock_med - real_med
            print(
                f"  {z_band} - {i_band}: real={real_med:+.3f}, mock={mock_med:+.3f}, "
                f"delta={delta_med:+.3f}  (n_real={real_ok.sum()}, n_mock={mock_ok.sum()})"
            )

    print("\nPer-band median offset and KS diagnostics:")
    print("  delta = median(mock_measured_mag of detected mocks) - median(real_obs_mag)")
    for fi, band in enumerate(FILTER_SHORT):
        real_mag = real_data["mag"][fi]
        mock_mag = mock_data["mag"][fi]

        real_ok = np.isfinite(real_mag)
        mock_ok = np.isfinite(mock_mag) & (mock_mag < NONDET_MAG - 0.5)
        if not np.any(real_ok) or not np.any(mock_ok):
            print(f"  {band:>10s}: insufficient valid data")
            continue

        real_med = float(np.nanmedian(real_mag[real_ok]))
        mock_med = float(np.nanmedian(mock_mag[mock_ok]))
        delta_med = mock_med - real_med
        ks = ks_2samp(real_mag[real_ok], mock_mag[mock_ok])
        print(
            f"  {band:>10s}: real={real_med:.3f}, mock={mock_med:.3f}, "
            f"delta={delta_med:+.3f}  KS={ks.statistic:.3f} p={ks.pvalue:.1e}"
            f"  (n_real={real_ok.sum()}, n_mock={mock_ok.sum()})"
        )


def apply_mag_calibration(real_data, mock_data):
    """
    Apply per-band median magnitude calibration to shift mocks onto the real data.

    delta_mag[fi] = median(detected mock mag) - median(detected real mag)
    mag_corrected  = mag_mock - delta_mag      (shifts mock towards real)
    """
    delta_mag = np.full(len(FILTER_SHORT), np.nan)
    mock_mag_cal = mock_data["mag"].copy()

    print("\nPer-band calibration (delta = median(mock_detected) - median(real)):")
    for fi, band in enumerate(FILTER_SHORT):
        real_mag = real_data["mag"][fi]
        mock_mag = mock_data["mag"][fi]

        real_ok = np.isfinite(real_mag)
        mock_det = np.isfinite(mock_mag) & (mock_mag < NONDET_MAG - 0.5)

        if not np.any(real_ok) or not np.any(mock_det):
            print(f"  {band:>10s}: skipped (insufficient valid data)")
            continue

        real_med = float(np.nanmedian(real_mag[real_ok]))
        mock_med = float(np.nanmedian(mock_mag[mock_det]))
        d = mock_med - real_med
        delta_mag[fi] = d

        # Apply ONLY to detected objects — leave NaN and non-det sentinel (99) untouched
        mock_mag_cal[fi][mock_det] = mock_mag[mock_det] - d
        print(
            f"  {band:>10s}: delta={d:+.3f} mag  "
            f"(mock_med={mock_med:.3f}, real_med={real_med:.3f})  applied: -({d:+.3f})"
            f"  n_det={mock_det.sum()}"
        )

    mock_data_cal = dict(mock_data)
    mock_data_cal["mag"] = mock_mag_cal
    return mock_data_cal, delta_mag


def debug_fsps_output_units_once():
    """Print FSPS spectrum scale for peraa=False vs peraa=True."""
    try:
        import fsps
    except Exception as exc:
        print(f"FSPS unit debug skipped: could not import fsps ({exc})")
        return

    try:
        sp = fsps.StellarPopulation(
            zcontinuous=1,
            sfh=3,
            dust_type=2,
        )

        # Minimal flat SFH just to probe FSPS output units/scales.
        t = np.linspace(0.001, 1.0, 200)
        sfh = np.full_like(t, 1e-3)
        sp.set_tabular_sfh(t, sfh)

        _, spec_per_hz = sp.get_spectrum(tage=1.0, peraa=False)
        _, spec_per_aa = sp.get_spectrum(tage=1.0, peraa=True)

        L_sun = 3.828e33
        spec_erg_hz = spec_per_hz * L_sun
        spec_erg_aa = spec_per_aa * L_sun

        print("FSPS unit debug:")
        print(
            "  peraa=False -> FSPS returns Lsun/Hz; "
            f"spec_erg range = [{np.nanmin(spec_erg_hz):.3e}, {np.nanmax(spec_erg_hz):.3e}]"
        )
        print(
            "  peraa=True  -> FSPS returns Lsun/A;  "
            f"spec_erg range = [{np.nanmin(spec_erg_aa):.3e}, {np.nanmax(spec_erg_aa):.3e}]"
        )
    except Exception as exc:
        print(f"FSPS unit debug failed: {exc}")


def debug_mass_flux_scaling_fixed_nuisance(model, target_masses=(9.0, 10.0, 11.0)):
    """
    Check mass scaling in two ways:
      (1) covariance diagnostics in loaded atlas
      (2) strict manual FSPS regeneration with fixed nuisance parameters

    Expects theta order:
      0=logM*, 2=logSFR, 3=tau, 4=t_i, 5=[M/H], 6=Av, 7=z
    """
    if model.theta is None or model.obs is None or len(model.theta) < 3:
        print("Mass-scaling debug skipped: model.theta/model.obs not available")
        return

    logm_all = np.asarray(model.theta[:, 0], dtype=float)
    logsfr_all = np.asarray(model.theta[:, 2], dtype=float)
    logssfr_all = logsfr_all - logm_all
    flux_med_all = np.nanmedian(mag_to_flux_ujy(model.obs), axis=1)
    good = (
        np.isfinite(logm_all)
        & np.isfinite(logssfr_all)
        & np.isfinite(flux_med_all)
        & (flux_med_all > 0)
    )

    print("Mass-scaling debug: covariance in loaded atlas")
    corr_m_ssfr = np.corrcoef(logm_all[good], logssfr_all[good])
    print("  corrcoef(logM, log_sSFR):")
    print(corr_m_ssfr)

    logf_all = np.log10(flux_med_all[good])
    corr_triplet = np.corrcoef(np.vstack([logm_all[good], logssfr_all[good], logf_all]))
    print("  corrcoef([logM, log_sSFR, log10(median_flux)]):")
    print(corr_triplet)

    X = np.column_stack([logm_all[good], logssfr_all[good], np.ones(good.sum())])
    beta, *_ = np.linalg.lstsq(X, logf_all, rcond=None)
    print(
        "  linear fit log10(flux)=a*logM + b*log_sSFR + c: "
        f"a={beta[0]:.3f}, b={beta[1]:.3f}, c={beta[2]:.3f}"
    )

    print("Mass-scaling debug: strict manual FSPS regeneration (fixed nuisance)")
    try:
        import fsps
        from astropy.cosmology import FlatLambdaCDM
    except Exception as exc:
        print(f"  manual FSPS debug skipped: {exc}")
        return

    tau_fix = float(np.nanmedian(model.theta[:, 3]))
    ti_fix = float(np.nanmedian(model.theta[:, 4]))
    met_fix = float(np.nanmedian(model.theta[:, 5]))
    av_fix = float(np.nanmedian(model.theta[:, 6]))
    z_fix = float(np.nanmedian(model.theta[:, 7]))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    age_gyr = float(cosmo.age(z_fix).value)
    t = np.linspace(0.001, age_gyr, 1000)
    print(f"  cosmic age: {cosmo.age(z_fix).value:.6f} Gyr")
    print(f"  sfh time max: {np.max(t):.6f} Gyr")

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=3, dust_type=2)
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1
    sp.params['dust2'] = av_fix
    sp.params['logzsol'] = met_fix
    sp.params['gas_logz'] = met_fix
    sp.params['zred'] = z_fix

    print(
        f"  fixed nuisance: z={z_fix:.3f}, Av={av_fix:.3f}, [M/H]={met_fix:.3f}, "
        f"tau={tau_fix:.3f}, ti={ti_fix:.3f}"
    )

    out = []
    for logm in target_masses:
        sfh_gyr, t_axis = sfh_delayed_exponential(t, logmassval=float(logm), tau=tau_fix, ti=ti_fix)
        sfh_yr = sfh_gyr / 1e9
        sfh_yr = np.where(np.isnan(sfh_yr) | (sfh_yr < 1e-33), 1e-33, sfh_yr)

        sp.set_tabular_sfh(t_axis, sfh_yr)
        _, spec = sp.get_spectrum(tage=age_gyr + 1e-4, peraa=False)
        flux_ujy = convert_to_microjansky(spec, z_fix, cosmo)
        med_flux = float(np.nanmedian(flux_ujy))

        recent_sfr = float(np.mean(sfh_yr[-100:]))
        logssfr = np.log10(max(recent_sfr, 1e-300)) - float(logm)
        out.append((float(logm), med_flux, logssfr))

    for logm, med_flux, logssfr in out:
        print(f"  logM={logm:.1f}, median_flux={med_flux:.6e} uJy, log_sSFR={logssfr:.3f}")

    logm = np.array([row[0] for row in out], dtype=float)
    logf = np.log10(np.array([max(row[1], 1e-300) for row in out], dtype=float))
    slope, intercept = np.polyfit(logm, logf, 1)
    print(f"  strict slope log10(flux) vs logM = {slope:.6f} (expected 1.0)")
    print(f"  strict intercept = {intercept:.6f}")

    strict_logssfr = np.array([row[2] for row in out], dtype=float)
    print("  strict corrcoef(logM, log_sSFR):")
    if np.nanstd(strict_logssfr) < 1e-12:
        print("  undefined (log_sSFR is fixed by construction; zero variance)")
    else:
        print(np.corrcoef(logm, strict_logssfr))


def debug_flux_scale(real_data, mock_data):
    """Print per-band median flux ratio: noisy mock (detected) vs real."""
    print("\n=== FLUX SCALE DEBUG (noisy mock detected vs real) ===")
    for fi, band in enumerate(FILTER_SHORT):
        real_flux = real_data["flux"][fi]
        mock_mag_noisy = mock_data["mag"][fi]
        mock_flux = mag_to_flux_ujy(mock_mag_noisy)

        real_ok = np.isfinite(real_flux) & (real_flux > 0)
        mock_ok = np.isfinite(mock_mag_noisy) & (mock_mag_noisy < NONDET_MAG - 0.5)
        if not np.any(real_ok) or not np.any(mock_ok):
            print(f"{band:>10s}: insufficient valid fluxes")
            continue

        real_med = np.nanmedian(real_flux[real_ok])
        mock_med = np.nanmedian(mock_flux[mock_ok])
        ratio = mock_med / real_med
        print(f"{band:>10s}: mock/real flux ratio = {ratio:.3f}")


def load_real_data(fits_path, patch_id=PATCH_ID, phot_type=None, snr_min=SNR_DETECTION_THRESHOLD):
    """
    Load photometry from COSMOS-Deep FITS catalog.

    Returns a dict with filter-major arrays.
    """
    from astropy.table import Table

    if phot_type is None:
        raise ValueError("phot_type must be provided explicitly")

    print(f"Loading real data from {fits_path}  (patch_id={patch_id}, phot_type={phot_type})")
    cat = Table.read(fits_path)
    # Patch selection
    patch_col = cat["patch_id_list"]
    try:
        mask = patch_col == int(patch_id)
    except (ValueError, TypeError):
        mask = np.zeros(len(cat), dtype=bool)
    str_mask = np.array([str(v).strip() == str(patch_id) for v in patch_col])
    mask = mask | str_mask
    cat = cat[mask]
    print(f"  {len(cat)} galaxies in patch {patch_id}")

    n_filt = len(FILTER_COL_STEMS)
    n_gal = len(cat)
    real_mag = np.full((n_filt, n_gal), np.nan)
    real_sigma = np.full((n_filt, n_gal), np.nan)
    real_flux = np.full((n_filt, n_gal), np.nan)
    real_err = np.full((n_filt, n_gal), np.nan)
    real_valid = np.zeros((n_filt, n_gal), dtype=bool)

    for fi, stem in enumerate(FILTER_COL_STEMS):
        # templfit uses a different column naming convention (no _aper suffix)
        if phot_type == "templfit":
            if stem == "vis":
                fcol = "flux_vis_psf"
                ecol = "fluxerr_vis_psf"
            else:
                fcol = f"flux_{stem}_templfit"
                ecol = f"fluxerr_{stem}_templfit"
        else:
            fcol = f"flux_{stem}_{phot_type}_aper"
            ecol = f"fluxerr_{stem}_{phot_type}_aper"
        if fcol not in cat.colnames:
            print(f"  WARNING: column {fcol!r} not found — filter {FILTER_SHORT[fi]} skipped")
            continue
        flux = np.asarray(cat[fcol], dtype=float)
        err = np.asarray(cat[ecol], dtype=float) if ecol in cat.colnames else np.full(n_gal, np.nan)

        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        snr = np.where(valid, flux / err, np.nan)
        detected = valid & np.isfinite(snr) & (snr >= snr_min) & (flux > 0)

        real_flux[fi] = flux
        real_err[fi] = err
        real_valid[fi] = valid
        real_mag[fi] = np.where(detected, flux_ujy_to_mag(flux), np.nan)
        real_sigma[fi] = np.where(detected, flux_ujy_to_mag_err(flux, err), np.nan)

    real_det = np.isfinite(real_mag) & np.isfinite(real_sigma)
    print(f"  Detection fractions: " +
          ", ".join(f"{FILTER_SHORT[i]}={real_det[i].mean():.2f}" for i in range(n_filt)))
    return {
        "mag": real_mag,
        "sigma": real_sigma,
        "flux": real_flux,
        "err": real_err,
        "valid": real_valid,
    }


def get_mock_arrays(model):
    """
    Extract mock (mag, sigma) from an sbipix instance after add_noise_nan_limit_all().

    Returns a dict with filter-major arrays.
    """
    # model.mag shape: (n_sim, n_filt, 2)
    # observation_space='mag'  -> [:,:,0]=mag,  [:,:,1]=sigma_mag
    # observation_space='flux' -> [:,:,0]=flux, [:,:,1]=sigma_flux
    obs_space = getattr(model, "noise_observation_space", "mag")
    first = model.mag[:, :, 0].T.copy()
    second = model.mag[:, :, 1].T.copy()

    if obs_space == "flux":
        mock_flux = first
        mock_sigma_flux = np.maximum(second, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            mock_mag = np.where(mock_flux > 0, flux_ujy_to_mag(mock_flux), NONDET_MAG)
            mock_sigma = np.where(
                mock_flux > 0,
                (2.5 / LN10) * np.abs(mock_sigma_flux / np.maximum(np.abs(mock_flux), 1e-12)),
                np.nan,
            )
    else:
        mock_mag = first
        mock_sigma = second
        with np.errstate(divide='ignore', invalid='ignore'):
            mock_flux = np.where(mock_mag < NONDET_MAG - 0.5, mag_to_flux_ujy(mock_mag), np.nan)
        mock_sigma_flux = np.full_like(mock_flux, np.nan, dtype=float)

    true_mag = model.obs.T.copy()
    true_flux = mag_to_flux_ujy(true_mag)
    indices = np.arange(model.obs.shape[0], dtype=int)
    return {
        "mag": mock_mag,
        "sigma": mock_sigma,
        "flux": mock_flux,
        "sigma_flux": mock_sigma_flux,
        "observation_space": obs_space,
        "true_mag": true_mag,
        "true_flux": true_flux,
        "indices": indices,
    }


def downsample_plot_data(data, max_objects, seed=0):
    """Return a shallow copy of plot data with at most max_objects galaxies."""
    if max_objects is None or max_objects <= 0:
        return data

    sample_size = None
    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == len(FILTER_SHORT):
            sample_size = value.shape[1]
            break
    if sample_size is None or sample_size <= max_objects:
        return data

    rng = np.random.default_rng(seed)
    selected_idx = np.sort(rng.choice(sample_size, size=max_objects, replace=False))
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.ndim >= 2 and value.shape[0] == len(FILTER_SHORT) and value.shape[1] == sample_size:
                out[key] = value[:, selected_idx, ...]
            elif value.ndim == 1 and value.shape[0] == sample_size:
                out[key] = value[selected_idx]
            else:
                out[key] = value
        else:
            out[key] = value
    out["indices"] = selected_idx
    return out


def run_flux_asinh_slope_diagnostic(model, mock_data):
    """Report slope(logM vs asinh(flux/softening)) before/after noise in z bins."""
    if getattr(model, "noise_observation_space", "mag") != "flux":
        return

    z_bins = [
        (0.00, 0.25, "z=[0.00,0.25)"),
        (0.25, 0.50, "z=[0.25,0.50)"),
        (0.50, 1.00, "z=[0.50,1.00)"),
        (1.00, 1.50, "z=[1.00,1.50)"),
        (1.50, 2.00, "z=[1.50,2.00)"),
        (2.00, 3.00, "z=[2.00,3.00)"),
        (3.00, 5.00, "z=[3.00,5.00)"),
    ]

    def _slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if np.sum(ok) < 20:
            return np.nan
        return float(np.polyfit(x[ok], y[ok], 1)[0])

    logm = np.asarray(model.theta[:, 0], dtype=float)
    z = np.asarray(model.theta[:, 7], dtype=float)
    before_flux = np.asarray(model.true_flux if hasattr(model, "true_flux") else mock_data["true_flux"], dtype=float)
    after_flux = np.asarray(mock_data["flux"], dtype=float)
    limits = np.asarray(model.limits, dtype=float)

    print("\nFlux-space slope diagnostic (asinh transform; includes negative noisy flux):")
    print("  band       z_bin            slope_before  slope_after   ratio")
    collapse = 0
    total = 0
    for bi, band in enumerate(FILTER_SHORT):
        soft = max(float(limits[bi]), 1e-12)
        yb = np.arcsinh(before_flux[bi] / soft)
        ya = np.arcsinh(after_flux[bi] / soft)
        for z0, z1, label in z_bins:
            m = (z >= z0) & (z < z1)
            sb = _slope(logm[m], yb[m])
            sa = _slope(logm[m], ya[m])
            if np.isfinite(sb) and np.isfinite(sa) and abs(sb) > 1e-12:
                ratio = sa / sb
                total += 1
                if ratio < 0.5:
                    collapse += 1
                print(f"  {band:<10} {label:<16} {sb:+.4f}      {sa:+.4f}      {ratio:+.3f}")
    if total > 0:
        print(f"  Collapse summary: {collapse}/{total} bins with ratio<0.5 ({100*collapse/total:.1f}%)")


def build_validation_model(args, obs_dir, library_dir):
    """Create and configure the sbipix model used for validation."""
    from sbipix import sbipix

    model = sbipix()
    noise_prefix = f"north_{args.phot_type}"
    limits_file = f"background_noise_{noise_prefix}.npy"
    model.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(obs_dir),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=limits_file,
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    model.atlas_path = str(library_dir) + "/"
    model.model_path = str(library_dir) + "/"
    model.atlas_name = "atlas_obs_euclid_north_validate"
    model.n_simulation = args.n_sim
    model.parametric = True
    model.both_masses = True
    model.infer_z = False
    model.include_limit = True
    model.include_sigma = True
    model.configure_noise_model(
        sigma_sampler=args.sigma_sampler,
        detection_model=args.detection_model,
        observation_space=args.observation_space,
    )
    # Enforce the SAME SNR detection threshold in mocks as used for real data selection
    model.snr_threshold = SNR_DETECTION_THRESHOLD
    return model, noise_prefix, limits_file


def load_or_simulate_model(model, args, outdir):
    """Load an existing atlas or simulate a new one, then save theta histograms."""
    if args.skip_simulate:
        print("[1/3] Loading existing simulation...")
        model.load_simulation()
    else:
        print(f"[1/3] Simulating {args.n_sim} galaxy SEDs...")
        model.simulate(**SIMULATION_CONFIG)
        model.load_simulation()

    plot_theta(
        model,
        save=True,
        filename=str(outdir / "theta_histograms.png"),
    )


def save_validation_plots(real_data, mock_data, model, outdir):
    # --- New: Flux ratio vs magnitude and flux-flux scatter plots ---
    for fi, band in enumerate(FILTER_SHORT):
        real_flux = real_data["flux"][fi]
        mock_mag_noisy = mock_data["mag"][fi]
        mock_flux = mag_to_flux_ujy(mock_mag_noisy)
        real_mag = real_data["mag"][fi]
        # Only use detected objects for both real and mock
        real_ok = np.isfinite(real_flux) & (real_flux > 0) & np.isfinite(real_mag)
        mock_ok = np.isfinite(mock_flux) & (mock_mag_noisy < NONDET_MAG - 0.5)
        if np.any(real_ok) and np.any(mock_ok):
            mag_bins = np.linspace(MAG_BRIGHT, MAG_FAINT, 40)
            plt.figure(figsize=(7, 5))
            bin_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
            median_real_flux = np.full_like(bin_centers, np.nan)
            median_mock_flux = np.full_like(bin_centers, np.nan)
            ratio = np.full_like(bin_centers, np.nan)
            for i, (lo, hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
                in_bin_real = real_ok & (real_mag >= lo) & (real_mag < hi)
                in_bin_mock = mock_ok & (mock_mag_noisy >= lo) & (mock_mag_noisy < hi)
                if np.any(in_bin_real):
                    median_real_flux[i] = np.nanmedian(real_flux[in_bin_real])
                if np.any(in_bin_mock):
                    median_mock_flux[i] = np.nanmedian(mock_flux[in_bin_mock])
                if np.isfinite(median_real_flux[i]) and np.isfinite(median_mock_flux[i]) and median_real_flux[i] > 0:
                    ratio[i] = median_mock_flux[i] / median_real_flux[i]
            plt.plot(bin_centers, ratio, marker='o', linestyle='-', color='navy')
            plt.xlabel(f"{band} magnitude")
            plt.ylabel("Median mock/real flux ratio")
            plt.title(f"Flux ratio vs magnitude: {band}")
            plt.ylim(0, np.nanmax(ratio)*1.2 if np.nanmax(ratio) > 0 else 2)
            plt.grid(True, alpha=0.3)
            out_flux_ratio = outdir / f"flux_ratio_vs_mag_{band.replace('-', '_')}.png"
            plt.tight_layout()
            plt.savefig(out_flux_ratio, dpi=170)
            plt.close()

            # Removed flux–flux scatter/hexbin plot as requested
    """Generate and save all validation plots."""
    def plot_sigma_model_curve(fi):
        if getattr(model, "noise_sigma_mag_params", None) is None:
            model._prepare_sigma_mag_lognormal_params()

        mgrid = np.linspace(20, 30, 100)
        interp_centers_all = getattr(model, "noise_sigma_mag_interp_centers", None)
        interp_means_all = getattr(model, "noise_sigma_mag_interp_means", None)
        sigma_floor = getattr(model, "noise_sigma_floor", 8e-3)

        centers = None if interp_centers_all is None else interp_centers_all[fi]
        means = None if interp_means_all is None else interp_means_all[fi]
        if centers is not None and means is not None and len(centers) >= 2:
            sigma_pred = np.interp(mgrid, centers, means)
            sigma_pred = np.maximum(sigma_pred, sigma_floor)
        else:
            coeffs = np.asarray(model.noise_sigma_mag_params[fi], dtype=float).ravel()
            if coeffs.size >= 4:
                a, b, c, _scatter = coeffs[:4]
            elif coeffs.size == 3:
                a, b, c = coeffs
            elif coeffs.size == 2:
                a, b = coeffs
                c = 0.0
            else:
                a = np.log(0.1)
                b = 0.0
                c = 0.0
            sigma_pred = np.exp(a + b * mgrid + c * mgrid * mgrid)
            sigma_pred = np.maximum(sigma_pred, sigma_floor)

        real_mag = real_data["mag"][fi]
        real_sigma = real_data["sigma"][fi]
        valid = np.isfinite(real_mag) & np.isfinite(real_sigma) & (real_sigma > 0)

        plt.figure(figsize=(6, 4.5))
        if np.any(valid):
            plt.scatter(real_mag[valid], real_sigma[valid], alpha=0.1, s=5, label='real')
        plt.plot(mgrid, sigma_pred, color='crimson', lw=2, label='model')
        plt.yscale('log')
        plt.xlabel('Magnitude')
        plt.ylabel('Sigma (mag)')
        plt.title(f"Sigma model curve: {FILTER_SHORT[fi]}")
        plt.legend()
        plt.tight_layout()
        out = outdir / f"sigma_model_curve_{FILTER_SHORT[fi].replace('-', '_')}.png"
        plt.savefig(out, dpi=170)
        plt.close()
        return out

    debug_flux_scale(real_data, mock_data)
    print_debug_diagnostics(real_data, mock_data)
    print_mock_plot_counts(mock_data)

    print(f"\nSaving validation plots to {outdir}/")
    saved = []

    out = plot_sigma_vs_mag_grid(
        real_data["mag"], real_data["sigma"], mock_data["mag"], mock_data["sigma"],
        model.mean_sigma_obs, model.percentiles, outdir,
        filter_short=FILTER_SHORT,
        nondet_mag=NONDET_MAG,
        mag_bright=MAG_BRIGHT,
        mag_faint=MAG_FAINT,
    )
    saved.append(out)
    print(f"  {out.name}")

    for fi in range(len(FILTER_SHORT)):
        out = plot_sigma_vs_mag(
            fi, real_data["mag"], real_data["sigma"], mock_data["mag"], mock_data["sigma"],
            model.mean_sigma_obs, model.percentiles, outdir,
            filter_short=FILTER_SHORT,
            nondet_mag=NONDET_MAG,
            mag_bright=MAG_BRIGHT,
            mag_faint=MAG_FAINT,
        )
        saved.append(out)

        out = plot_mag_histogram(
            fi, real_data["mag"], mock_data["mag"], outdir,
            filter_short=FILTER_SHORT,
            nondet_mag=NONDET_MAG,
            mag_bright=MAG_BRIGHT,
            mag_faint=MAG_FAINT,
        )
        if out:
            saved.append(out)

        out = plot_true_mag_histogram(
            fi, real_data["mag"], mock_data["true_mag"], outdir,
            filter_short=FILTER_SHORT,
            mag_bright=MAG_BRIGHT,
            mag_faint=MAG_FAINT,
        )
        if out:
            saved.append(out)

        out = plot_detection_fraction_vs_flux(
            fi, real_data, mock_data, model.limits, outdir,
            filter_short=FILTER_SHORT,
            snr_detection_threshold=SNR_DETECTION_THRESHOLD,
            nondet_mag=NONDET_MAG,
        )
        if out:
            saved.append(out)

        out = plot_sigma_distribution(
            fi, real_data["sigma"], mock_data["sigma"], outdir,
            filter_short=FILTER_SHORT,
        )
        if out:
            saved.append(out)

        out = plot_sigma_model_curve(fi)
        if out:
            saved.append(out)

    print("  [per-filter above: sigma_vs_mag, mag_hist, det_fraction, sigma_dist]")

    out = plot_colors(
        real_data["mag"], mock_data["mag"], outdir,
        filter_short=FILTER_SHORT,
        color_pairs=COLOR_PAIRS,
        nondet_mag=NONDET_MAG,
    )
    saved.append(out)
    print(f"  {out.name}")

    out = plot_intrinsic_color_color(real_data, mock_data, outdir, FILTER_SHORT)
    if out:
        saved.append(out)
        print(f"  {out.name}")

    param_plots = plot_intrinsic_color_vs_parameters(mock_data, model, outdir, FILTER_SHORT)
    saved.extend(param_plots)
    for out in param_plots:
        print(f"  {out.name}")

    print(f"\nDone — {len(saved)} plots saved to {outdir}/")
    return saved
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Pre-training validation: compare mock noise model to real COSMOS data"
    )
    p.add_argument("--phot-type", default="3fwhm",
                   choices=["2fwhm", "3fwhm", "templfit"],
                   help="Photometry type to use for both real COSMOS photometry and matching noise products (default: 3fwhm)")
    p.add_argument("--n-sim", type=int, default=10000,
                   help="Number of mock simulations (default: 10000; use smaller only for local smoke tests)")
    p.add_argument("--outdir", default="sbi-logs/validate",
                   help="Output directory for plots (default: sbi-logs/validate)")
    p.add_argument("--skip-simulate", "--skip-sim", action="store_true",
                   help="Skip simulation step; load existing atlas instead")
    p.add_argument("--sigma-sampler", choices=["empirical", "truncnorm", "mag_lognormal"], default="empirical",
                   help="Distribution used to sample sigma values (default: empirical; mag_lognormal fits log(sigma)=a+b*mag per band)")
    p.add_argument("--detection-model", choices=["hard", "probabilistic"], default="probabilistic",
                   help="Detection model after noise injection: hard flux threshold or smooth S/N transition (default: probabilistic)")
    p.add_argument("--observation-space", choices=["mag", "flux"], default="mag",
                   help="Noise model output space: mag (legacy) or flux (keeps negative noisy realizations)")
    p.add_argument("--mock-match", choices=["none", "vis_yj2d"], default="none",
                   help="Reweight/resample mocks to match real observed distributions (default: none)")
    p.add_argument("--calibrate", action="store_true",
                   help="Apply per-band median magnitude calibration after mock-matching: "
                        "mag_corrected = mag_mock - delta where delta = median(detected mock) - median(real)")
    p.add_argument("--fast", action="store_true",
                   help="Fast local mode: cap n_sim and downsample heavy plotting inputs")
    p.add_argument("--max-plot-objects", type=int, default=None,
                   help="Maximum number of real/mock objects passed to plotting routines")
    return p


def main():
    args = build_parser().parse_args()

    if args.fast:
        args.n_sim = min(args.n_sim, 5000)
        if args.max_plot_objects is None:
            args.max_plot_objects = 15000

    if args.n_sim < 10000:
        print(f"NOTE: n_sim={args.n_sim} is fine for a local smoke test, but ~10000+ is recommended for validation.")
        print("      Use the server for the final comparison plots.")

    # ------------------------------------------------------------------
    # Setup paths
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    obs_dir = project_root / "obs" / "obs_properties"
    library_dir = project_root / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fits_path = CATALOG_FILE
    if not os.path.isabs(fits_path):
        fits_path = str(project_root / fits_path)

    model, noise_prefix, limits_file = build_validation_model(args, obs_dir, library_dir)

    if args.sigma_sampler == "mag_lognormal" and not hasattr(model, "_sample_sigma_mag_lognormal"):
        raise RuntimeError(
            "Requested --sigma-sampler mag_lognormal, but loaded sbipix does not support it. "
            "Ensure you run from this repo and that local src/ is used (or reinstall package in editable mode)."
        )

    print("Noise-model settings:")
    print(f"  noise prefix   = {noise_prefix}")
    print(f"  limits file    = {limits_file}")
    print(f"  sigma sampler  = {model.noise_sigma_sampler}")
    print(f"  detect model   = {model.noise_detection_model}")
    print(f"  obs space      = {model.noise_observation_space}")
    print(f"  mock match     = {args.mock_match}")
    print(f"  calibrate      = {'on' if args.calibrate else 'off'}")
    print(f"  fast mode      = {'on' if args.fast else 'off'}")
    print(f"  max plot objs  = {args.max_plot_objects if args.max_plot_objects is not None else 'all'}")
    print(f"  det SNR cut    = {SNR_DETECTION_THRESHOLD}")

    np.random.seed(0)

    load_or_simulate_model(model, args, outdir)

    debug_i = 0
    debug_band = 0
    debug_band_name = FILTER_SHORT[debug_band] if debug_band < len(FILTER_SHORT) else f"band_{debug_band}"
    raw_flux_debug = np.nan
    if model.obs is not None and len(model.obs) > debug_i and model.obs.shape[1] > debug_band:
        raw_mag_debug = float(model.obs[debug_i, debug_band])
        raw_flux_debug = float(mag_to_flux_ujy(raw_mag_debug))
        print("\n=== DEBUG: RAW ATLAS FLUX ===")
        print(f"  galaxy={debug_i}, band={debug_band} ({debug_band_name})")
        print(f"  Atlas flux (pre-noise): {raw_flux_debug:.3e} uJy")

    debug_fsps_output_units_once()

    if model.theta is not None and model.obs is not None and len(model.theta) >= 2:
        print("Checking luminosity scaling...")
        # model.obs stores noiseless magnitudes (n_sim, n_filt); convert to flux for comparison
        m1 = model.theta[0, 0]
        m2 = model.theta[1, 0]
        f1 = float(np.nanmedian(mag_to_flux_ujy(model.obs[0])))
        f2 = float(np.nanmedian(mag_to_flux_ujy(model.obs[1])))
        print(f"  logM1={m1:.2f}, median_flux1={f1:.3e} uJy")
        print(f"  logM2={m2:.2f}, median_flux2={f2:.3e} uJy")
        if f1 > 0 and f2 > 0:
            print(f"  flux ratio = {f2/f1:.2f}, expected ~ {10**(m2-m1):.2f}")
        else:
            print("  flux ratio: skipped (non-positive flux)")

        debug_mass_flux_scaling_fixed_nuisance(model, target_masses=(9.0, 10.0, 11.0))
    else:
        print("Checking luminosity scaling... skipped (theta/obs unavailable)")

    # ------------------------------------------------------------------
    # Mask: keep galaxies that are potentially detectable in at least one
    # band, using each band's own 1-sigma depth limit.
    # ------------------------------------------------------------------
    obs_flux = mag_to_flux_ujy(model.obs)                     # (n_sim, n_filt)
    _limits_arr = np.load(str(obs_dir / limits_file))
    n_filt_used = model.obs.shape[1]
    sigma_lim_per_filter = np.asarray(_limits_arr[:n_filt_used], dtype=float)  # (n_filt,)
    # Keep only objects with true-flux SNR >= detection threshold in at least
    # one band.  This matches the survey selection function and prevents
    # training on intrinsically undetectable galaxies that would bias the
    # posterior towards non-detections (particularly at high-z).
    snr_in_band = obs_flux / np.maximum(sigma_lim_per_filter[None, :], 1e-12)
    bright_in_band = snr_in_band >= SNR_DETECTION_THRESHOLD
    flux_mask = np.any(bright_in_band, axis=1)
    n_before = model.n_simulation
    model.theta = model.theta[flux_mask]
    model.obs   = model.obs[flux_mask]
    model.n_simulation = int(flux_mask.sum())
    n_kept_per_filter = np.sum(bright_in_band, axis=0)
    lim_txt = ", ".join(f"{x:.4f}" for x in sigma_lim_per_filter)
    kept_txt = ", ".join(str(int(x)) for x in n_kept_per_filter)
    print(
        f"  SNR mask (any band true-flux/sigma_lim >= {SNR_DETECTION_THRESHOLD:.1f}): "
        f"{model.n_simulation} / {n_before} galaxies kept"
    )
    print(f"  per-filter sigma_lim [μJy]: [{lim_txt}]")
    print(f"  per-filter N(true SNR >= {SNR_DETECTION_THRESHOLD:.1f}): [{kept_txt}]")

    print("[2/3] Applying observational realism...")
    model.load_obs_features()
    model.add_noise_nan_limit_all()

    if model.mag is not None and len(model.mag) > debug_i and model.mag.shape[1] > debug_band:
        noisy_mag_debug = float(model.mag[debug_i, debug_band, 0])
        noisy_flux_debug = float(mag_to_flux_ujy(noisy_mag_debug))
        print("\n=== DEBUG: AFTER NOISE ===")
        print(f"  galaxy={debug_i}, band={debug_band} ({debug_band_name})")
        print(f"  Noisy flux: {noisy_flux_debug:.3e} uJy")
        if np.isfinite(raw_flux_debug) and raw_flux_debug > 0:
            print(f"  Noisy/Raw flux ratio: {noisy_flux_debug/raw_flux_debug:.3f}")

    # Clean up NaN thetas
    ok = np.isfinite(np.sum(model.theta, axis=1))
    model.theta = model.theta[ok]
    model.mag = model.mag[ok]
    model.obs = model.obs[ok]
    model.n_simulation = len(model.theta)
    print(f"  {model.n_simulation} valid mock galaxies")

    # ------------------------------------------------------------------
    # Load real data
    # ------------------------------------------------------------------
    print("[3/3] Loading real COSMOS data...")
    real_data = load_real_data(
        fits_path, patch_id=PATCH_ID, phot_type=args.phot_type
    )

    # ------------------------------------------------------------------
    # Extract mock arrays  +  noiseless SED mags
    # ------------------------------------------------------------------
    mock_data = get_mock_arrays(model)
    run_flux_asinh_slope_diagnostic(model, mock_data)

    # Single-band median mock/real flux check (requested quick ratio sanity)
    real_flux_band = real_data["flux"][debug_band]
    mock_mag_band = mock_data["mag"][debug_band]
    mock_flux_band = mag_to_flux_ujy(mock_mag_band)
    real_ok = np.isfinite(real_flux_band) & (real_flux_band > 0)
    mock_ok = np.isfinite(mock_mag_band) & (mock_mag_band < NONDET_MAG - 0.5)
    if np.any(real_ok) and np.any(mock_ok):
        real_med = float(np.nanmedian(real_flux_band[real_ok]))
        mock_med = float(np.nanmedian(mock_flux_band[mock_ok]))
        ratio = mock_med / real_med if real_med > 0 else np.nan
        print("\n=== DEBUG: MOCK VS REAL FLUX (single band) ===")
        print(f"  band={debug_band} ({debug_band_name})")
        print(f"  median real flux: {real_med:.3e} uJy")
        print(f"  median mock flux: {mock_med:.3e} uJy")
        print(f"  mock/real ratio:  {ratio:.3f}")

    # ------------------------------------------------------------------
    # Optional mock reweighting/resampling to match the real observed prior
    # ------------------------------------------------------------------
    if args.mock_match != "none":
        mock_weights, match_msg = compute_mock_match_weights(real_data, mock_data)
        print(match_msg)
        print(f"  non-zero weights: {(mock_weights > 0).sum()} / {mock_weights.size}")
        mock_data, resample_msg = resample_mock_catalogue(mock_data, mock_weights, seed=0)
        print(f"  {resample_msg}")

    if args.calibrate:
        mock_data, delta_mag = apply_mag_calibration(real_data, mock_data)
        print(f"  Per-band calibration applied to {np.sum(np.isfinite(delta_mag))} / {len(FILTER_SHORT)} bands")

    plot_real_data = downsample_plot_data(real_data, args.max_plot_objects, seed=0)
    plot_mock_data = downsample_plot_data(mock_data, args.max_plot_objects, seed=1)
    if plot_real_data is not real_data or plot_mock_data is not mock_data:
        print(
            f"  Plot downsampling enabled: real={plot_real_data['mag'].shape[1]}, "
            f"mock={plot_mock_data['mag'].shape[1]}"
        )

    save_validation_plots(plot_real_data, plot_mock_data, model, outdir)

if __name__ == "__main__":
    main()
