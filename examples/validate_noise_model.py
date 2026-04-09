"""
validate_noise_model.py — Pre-training data-space validation.

Runs sbipix simulate + noise injection, loads real COSMOS-Deep photometry
(selected aperture), and produces side-by-side diagnostic plots to confirm
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
        --aperture 2fwhm \
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
    flux_ujy_to_mag,
    flux_ujy_to_mag_err,
    mag_to_flux_ujy,
    load_filter_metadata,
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
SNR_DETECTION_THRESHOLD = 2.0
MAG_BRIGHT = 16.0
MAG_FAINT = 30.0
CATALOG_FILE = "obs/obs_properties/COSMOS_DEEP.fits"
PATCH_ID = 98
AB_ZEROPOINT_JY = 3631.0
AB_ZEROPOINT_UJY = 3631e6
SIMULATION_CONFIG = {
    "mass_min": 6.0,
    "mass_max": 11.5,
    "sfr_prior_type": "sSFRlognormal",
    "ssfr_min": -12.5,
    "ssfr_max": -8.5,
    "z_prior": "flat",
    "z_min": 0.0,
    "z_max": 5.0,
    "Z_min": -0.8,
    "Z_max": 0.3,
    "dust_model": "Calzetti",
    "dust_prior": "exp",
    "Av_min": 0.2,
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
    mock_ok  = (
        np.isfinite(mock_vis) & (mock_vis < NONDET_MAG - 0.5)
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
        mock_ok = np.isfinite(mock_mag) & (mock_mag < NONDET_MAG - 0.5)

        if not np.any(real_ok) or not np.any(mock_ok):
            print(f"  {band:>10s}: skipped (insufficient valid data)")
            continue

        real_med = float(np.nanmedian(real_mag[real_ok]))
        mock_med = float(np.nanmedian(mock_mag[mock_ok]))
        d = mock_med - real_med
        delta_mag[fi] = d

        mock_mag_cal[fi] = mock_mag - d
        print(
            f"  {band:>10s}: delta={d:+.3f} mag  "
            f"(mock_med={mock_med:.3f}, real_med={real_med:.3f})  applied: -({d:+.3f})"
        )

    mock_data_cal = dict(mock_data)
    mock_data_cal["mag"] = mock_mag_cal
    return mock_data_cal, delta_mag


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


def load_real_data(fits_path, patch_id=PATCH_ID, aperture=None, snr_min=SNR_DETECTION_THRESHOLD):
    """
    Load photometry from COSMOS-Deep FITS catalog.

    Returns a dict with filter-major arrays.
    """
    from astropy.table import Table

    if aperture is None:
        raise ValueError("aperture must be provided explicitly")

    print(f"Loading real data from {fits_path}  (patch_id={patch_id}, aperture={aperture})")
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
        fcol = f"flux_{stem}_{aperture}_aper"
        ecol = f"fluxerr_{stem}_{aperture}_aper"
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
    # model.mag shape: (n_sim, n_filt, 2) — [:, :, 0]=mag, [:, :, 1]=sigma
    mock_mag = model.mag[:, :, 0].T.copy()     # (n_filt, n_sim)
    mock_sigma = model.mag[:, :, 1].T.copy()   # (n_filt, n_sim)
    true_mag = model.obs.T.copy()
    true_flux = mag_to_flux_ujy(true_mag)
    indices = np.arange(model.obs.shape[0], dtype=int)
    return {
        "mag": mock_mag,
        "sigma": mock_sigma,
        "true_mag": true_mag,
        "true_flux": true_flux,
        "indices": indices,
    }


def build_validation_model(args, obs_dir, library_dir):
    """Create and configure the sbipix model used for validation."""
    from sbipix import sbipix

    model = sbipix()
    noise_prefix = f"north_{args.aperture}"
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
    model.model_name = "post_obs_euclid_north_validate.pkl"
    model.n_simulation = args.n_sim
    model.parametric = True
    model.both_masses = True
    model.infer_z = False
    model.include_limit = True
    model.include_sigma = True
    model.configure_noise_model(
        sigma_sampler=args.sigma_sampler,
        detection_model=args.detection_model,
    )
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
    """Generate and save all validation plots."""
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
    p.add_argument("--aperture", default="3fwhm",
                   choices=["2fwhm", "3fwhm"],
                   help="Aperture to use for both real COSMOS photometry and matching noise products (default: 3fwhm)")
    p.add_argument("--n-sim", type=int, default=10000,
                   help="Number of mock simulations (default: 10000; use smaller only for local smoke tests)")
    p.add_argument("--outdir", default="sbi-logs/validate",
                   help="Output directory for plots (default: sbi-logs/validate)")
    p.add_argument("--skip-simulate", "--skip-sim", action="store_true",
                   help="Skip simulation step; load existing atlas instead")
    p.add_argument("--sigma-sampler", choices=["empirical", "truncnorm", "lognormal", "mag_lognormal"], default="empirical",
                   help="Distribution used to sample sigma values (default: empirical; mag_lognormal fits log(sigma)=a+b*mag per band)")
    p.add_argument("--detection-model", choices=["hard", "probabilistic"], default="probabilistic",
                   help="Detection model after noise injection: hard flux threshold or smooth S/N transition (default: probabilistic)")
    p.add_argument("--mock-match", choices=["none", "vis_yj2d"], default="vis_yj2d",
                   help="Reweight/resample mocks to match real observed distributions (default: vis_yj2d = 2D VIS×(Y-J) histogram)")
    p.add_argument("--calibrate", action="store_true",
                   help="Apply per-band median magnitude calibration after mock-matching: "
                        "mag_corrected = mag_mock - delta where delta = median(detected mock) - median(real)")
    return p


def main():
    args = build_parser().parse_args()

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
    print(f"  det SNR cut    = {SNR_DETECTION_THRESHOLD}")

    np.random.seed(0)

    load_or_simulate_model(model, args, outdir)

    print("[2/3] Applying observational realism...")
    model.load_obs_features()
    model.add_noise_nan_limit_all()

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
        fits_path, patch_id=PATCH_ID, aperture=args.aperture
    )

    # ------------------------------------------------------------------
    # Extract mock arrays  +  noiseless SED mags
    # ------------------------------------------------------------------
    mock_data = get_mock_arrays(model)

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

    save_validation_plots(real_data, mock_data, model, outdir)

if __name__ == "__main__":
    main()
