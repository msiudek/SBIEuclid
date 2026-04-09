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
        --mock-match vis1d \
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
)
from sbipix.plotting.diagnostics import plot_theta

import numpy as np
import matplotlib.pyplot as plt


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


def _ratio_weights(real_hist, mock_hist):
    """Return histogram-ratio weights using normalized frequencies."""
    real_hist = np.asarray(real_hist, dtype=float)
    mock_hist = np.asarray(mock_hist, dtype=float)
    real_total = real_hist.sum()
    mock_total = mock_hist.sum()
    if real_total <= 0 or mock_total <= 0:
        return np.zeros_like(mock_hist, dtype=float)
    real_pdf = real_hist / real_total
    mock_pdf = mock_hist / mock_total
    weights = np.zeros_like(mock_pdf, dtype=float)
    valid = mock_pdf > 0
    weights[valid] = real_pdf[valid] / mock_pdf[valid]
    return weights


def compute_mock_match_weights(real_data, mock_data):
    """Compute per-mock-object weights from the 1D VIS magnitude histogram."""
    n_mock = mock_data["mag"].shape[1]
    weights = np.ones(n_mock, dtype=float)

    band_idx = FILTER_SHORT.index("VIS")
    real_band = real_data["mag"][band_idx]
    mock_band = mock_data["mag"][band_idx]
    real_band_ok = np.isfinite(real_band)
    mock_band_ok = np.isfinite(mock_band) & (mock_band < NONDET_MAG - 0.5)

    bins_mag = np.linspace(MAG_BRIGHT, MAG_FAINT, 25)
    real_hist, _ = np.histogram(real_band[real_band_ok], bins=bins_mag)
    mock_hist, _ = np.histogram(mock_band[mock_band_ok], bins=bins_mag)
    bin_weights = _ratio_weights(real_hist, mock_hist)
    bin_idx = np.digitize(mock_band, bins_mag) - 1
    in_range = mock_band_ok & (bin_idx >= 0) & (bin_idx < len(bin_weights))
    weights[:] = 0.0
    weights[in_range] = bin_weights[bin_idx[in_range]]
    return weights, (
        f"mock matching: 1D VIS histogram with 24 bins "
        f"(real n={real_band_ok.sum()}, mock detected n={mock_band_ok.sum()})"
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
        if isinstance(value, np.ndarray) and value.shape[-1] == n_obj:
            out[key] = value[..., draw_idx]
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

    print("\nPer-band median offset diagnostics:")
    print("  delta = median(mock_true_mag of detected mocks) - median(real_obs_mag)")
    for fi, band in enumerate(FILTER_SHORT):
        real_mag = real_data["mag"][fi]
        mock_true_mag = mock_data["true_mag"][fi]
        mock_measured_mag = mock_data["mag"][fi]

        real_ok = np.isfinite(real_mag)
        mock_ok = np.isfinite(mock_true_mag) & np.isfinite(mock_measured_mag) & (mock_measured_mag < NONDET_MAG - 0.5)
        if not np.any(real_ok) or not np.any(mock_ok):
            print(f"  {band:>10s}: insufficient valid data")
            continue

        real_med = float(np.nanmedian(real_mag[real_ok]))
        mock_med = float(np.nanmedian(mock_true_mag[mock_ok]))
        delta_med = mock_med - real_med
        print(
            f"  {band:>10s}: real={real_med:.3f}, mock_true={mock_med:.3f}, "
            f"delta={delta_med:+.3f}  (n_real={real_ok.sum()}, n_mock={mock_ok.sum()})"
        )


def debug_flux_scale(real_data, mock_data):
    """Print per-band median flux ratio mock_true / real."""
    print("\n=== FLUX SCALE DEBUG ===")
    for fi, band in enumerate(FILTER_SHORT):
        real_flux = real_data["flux"][fi]
        mock_flux = mock_data["true_flux"][fi]

        real_ok = np.isfinite(real_flux) & (real_flux > 0)
        mock_ok = np.isfinite(mock_flux) & (mock_flux > 0)
        if not np.any(real_ok) or not np.any(mock_ok):
            print(f"{band:>10s}: insufficient valid fluxes")
            continue

        real_med = np.nanmedian(real_flux[real_ok])
        mock_med = np.nanmedian(mock_flux[mock_ok])
        ratio = mock_med / real_med
        print(f"{band:>10s}: mock/real flux ratio = {ratio:.3f}")


def _pair_color(mag_data, idx_a, idx_b):
    """Return color mag[idx_a] - mag[idx_b] and valid mask."""
    mag_a = mag_data[idx_a]
    mag_b = mag_data[idx_b]
    valid = np.isfinite(mag_a) & np.isfinite(mag_b)
    return mag_a - mag_b, valid


def plot_intrinsic_color_color(real_data, mock_data, outdir):
    """Compare observed and intrinsic (pre-noise) color-color loci."""
    idx = {name: FILTER_SHORT.index(name) for name in FILTER_SHORT}
    required = ["VIS", "NISP-Y", "NISP-J", "NISP-H"]
    if not all(name in idx for name in required):
        return None

    vis_y_real, m1r = _pair_color(real_data["mag"], idx["VIS"], idx["NISP-Y"])
    y_j_real, m2r = _pair_color(real_data["mag"], idx["NISP-Y"], idx["NISP-J"])
    j_h_real, m3r = _pair_color(real_data["mag"], idx["NISP-J"], idx["NISP-H"])

    vis_y_mock, m1m = _pair_color(mock_data["true_mag"], idx["VIS"], idx["NISP-Y"])
    y_j_mock, m2m = _pair_color(mock_data["true_mag"], idx["NISP-Y"], idx["NISP-J"])
    j_h_mock, m3m = _pair_color(mock_data["true_mag"], idx["NISP-J"], idx["NISP-H"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)

    mask_real = m1r & m2r
    mask_mock = m1m & m2m
    axes[0].scatter(vis_y_real[mask_real], y_j_real[mask_real], s=4, alpha=0.18, label="Real obs")
    axes[0].scatter(vis_y_mock[mask_mock], y_j_mock[mask_mock], s=4, alpha=0.18, label="Mock true")
    axes[0].set_xlabel("VIS - Y")
    axes[0].set_ylabel("Y - J")
    axes[0].legend(loc="best", fontsize=8)

    mask_real = m2r & m3r
    mask_mock = m2m & m3m
    axes[1].scatter(y_j_real[mask_real], j_h_real[mask_real], s=4, alpha=0.18, label="Real obs")
    axes[1].scatter(y_j_mock[mask_mock], j_h_mock[mask_mock], s=4, alpha=0.18, label="Mock true")
    axes[1].set_xlabel("Y - J")
    axes[1].set_ylabel("J - H")
    axes[1].legend(loc="best", fontsize=8)

    fig.suptitle("Intrinsic color-color (mock true) vs observed colors")
    fig.tight_layout()
    out = outdir / "colors_intrinsic_true_vs_real.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_intrinsic_color_vs_parameters(mock_data, model, outdir):
    """Plot intrinsic color versus z, Av, Z and sSFR."""
    if model.theta is None:
        return []

    idx = {name: FILTER_SHORT.index(name) for name in FILTER_SHORT}
    if "DECam-i" not in idx:
        return []

    theta_idx = mock_data.get("indices", None)
    if theta_idx is None:
        theta_idx = np.arange(mock_data["true_mag"].shape[1])
    theta_use = model.theta[np.asarray(theta_idx, dtype=int)]

    log_mstar = theta_use[:, 0]
    log_sfr = theta_use[:, 2]
    z_vals = theta_use[:, 7]
    av_vals = theta_use[:, 6]
    zmet_vals = theta_use[:, 5]
    ssfr_vals = log_sfr - log_mstar

    param_defs = [
        (z_vals, "z"),
        (av_vals, "Av"),
        (zmet_vals, "Z [M/H]"),
        (ssfr_vals, "log sSFR"),
    ]

    saved = []
    i_idx = idx["DECam-i"]
    for z_band in ["DECam-z", "HSC-z"]:
        if z_band not in idx:
            continue
        z_idx = idx[z_band]
        color = mock_data["true_mag"][z_idx] - mock_data["true_mag"][i_idx]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=140)
        for ax, (param_vals, param_label) in zip(axes.ravel(), param_defs):
            valid = np.isfinite(color) & np.isfinite(param_vals)
            ax.scatter(param_vals[valid], color[valid], s=4, alpha=0.18)
            ax.set_xlabel(param_label)
            ax.set_ylabel(f"{z_band} - DECam-i")
            ax.grid(alpha=0.2)

        fig.suptitle(f"Intrinsic color vs physical parameters: {z_band} - DECam-i")
        fig.tight_layout()
        out = outdir / f"intrinsic_color_vs_params_{z_band.replace('-', '_')}_minus_DECam_i.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)

    return saved


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

    out = plot_intrinsic_color_color(real_data, mock_data, outdir)
    if out:
        saved.append(out)
        print(f"  {out.name}")

    param_plots = plot_intrinsic_color_vs_parameters(mock_data, model, outdir)
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
    p.add_argument("--mock-match", choices=["none", "vis1d"], default="vis1d",
                   help="Reweight/resample mocks to match real observed distributions (default: vis1d)")
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

    save_validation_plots(real_data, mock_data, model, outdir)

if __name__ == "__main__":
    main()
