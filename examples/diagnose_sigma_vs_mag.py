"""
diagnose_sigma_vs_mag.py — Atlas noise/magnitude diagnostic.

Loads a dbatlas, injects observational noise, applies a detection filter,
and produces diagnostic plots comparing mock vs real catalogs.

Plots saved to --outdir:
    sigma_vs_mag_VIS.png   — σ vs VIS mag: mock scatter + model curve + real data
    sigma_vs_mag_NISP_H.png
    sigma_vs_z.png         — per-filter median σ vs redshift
    ssfr_vs_z.png          — sSFR vs redshift: real catalogs vs mock (all + detected)
    coverage.png           — detection fraction vs z
    sigma_grid.png         — 2×5 grid of σ vs mag for all 10 bands
    mag_vs_z.png           — VIS magnitude vs redshift
    mag_grid.png           — 2×5 grid of magnitude distributions

Usage
-----
python examples/diagnose_sigma_vs_mag.py \
    --atlas atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas \
    --phot-type templfit \
    --catalogs cosmos_deep cosmos_web desi \
    --min-det-bands 3 \
    --outdir sbi-logs/diagnose_v1.0 \
    2>&1 | tee sbi-logs/diagnose_v1.0.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbipix.utils.sed_utils import load_filter_metadata, flux_ujy_to_mag

ROOT    = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"

FILTER_META  = load_filter_metadata("filters_to_use.dat", filt_dir=str(OBS_DIR))
FILTER_SHORT = [m["short"]    for m in FILTER_META]
FILTER_STEMS = [m["col_stem"] for m in FILTER_META]
N_FILT = len(FILTER_META)

# Catalog paths and reference columns
# sfr_log=False means the SFR column is in linear units (M☉/yr) and needs log10
CATALOG_INFO = {
    "cosmos_deep": {
        "path": OBS_DIR / "COSMOS_DEEP_PHZ.fits",
        "z_col":    "PHZ_PP_MEDIAN_REDSHIFT",
        "mass_col": "PHZ_PP_MEDIAN_STELLARMASS",   # already log10
        "sfr_col":  "PHZ_PP_MEDIAN_SFR",           # linear M☉/yr → log10 below
        "sfr_log":  False,
    },
    "cosmos_web": {
        "path": OBS_DIR / "COSMOS-Web" / "matched_euclid_cosmosweb.fits",
        "z_col":    "z_lephare",
        "mass_col": "logM_lephare",
        "sfr_col":  "logSFR_lephare",
        "sfr_log":  True,
    },
    "desi": {
        "path": OBS_DIR / "COSMOS-DESI" / "matched_euclid_desi.fits",
        "z_col":    "z_desi",
        "mass_col": "logM_desi_Cigale",
        "sfr_col":  "logSFR_desi_Cigale",
        "sfr_log":  True,
    },
}

CAT_COLORS = {"cosmos_deep": "#1f77b4", "cosmos_web": "#ff7f0e", "desi": "#2ca02c"}
CAT_LABELS = {"cosmos_deep": "COSMOS-Deep", "cosmos_web": "COSMOS-Web", "desi": "COSMOS-DESI"}


# ── helpers ────────────────────────────────────────────────────────────────

def build_phot_col(stem, phot_type, err=False):
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"
    return f"{prefix}_{stem}_{phot_type}_aper"


def load_catalog(cat_key, phot_type):
    """Load a catalog and return flux, fluxerr (n_gal, n_filt), z, logM, logSFR."""
    from astropy.table import Table
    info = CATALOG_INFO[cat_key]
    cat = Table.read(info["path"])
    print(f"  [{cat_key}] Loading: {info['path'].name}")

    z = np.array(cat[info["z_col"]], dtype=float) if info["z_col"] in cat.colnames else np.full(len(cat), np.nan)

    logM = np.full(len(cat), np.nan)
    if info["mass_col"] in cat.colnames:
        logM = np.array(cat[info["mass_col"]], dtype=float)

    logSFR = np.full(len(cat), np.nan)
    if info["sfr_col"] in cat.colnames:
        raw_sfr = np.array(cat[info["sfr_col"]], dtype=float)
        if info.get("sfr_log", True):
            logSFR = raw_sfr
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                logSFR = np.where(raw_sfr > 0, np.log10(raw_sfr), np.nan)

    flux_cols    = [build_phot_col(s, phot_type, err=False) for s in FILTER_STEMS]
    fluxerr_cols = [build_phot_col(s, phot_type, err=True)  for s in FILTER_STEMS]

    available_f  = [c for c in flux_cols    if c in cat.colnames]
    available_fe = [c for c in fluxerr_cols if c in cat.colnames]

    if len(available_f) < N_FILT or len(available_fe) < N_FILT:
        print(f"    WARNING: only {len(available_f)}/{N_FILT} flux cols available — skipping flux load")
        return None

    flux    = np.column_stack([np.array(cat[c], dtype=float) for c in flux_cols])
    fluxerr = np.column_stack([np.array(cat[c], dtype=float) for c in fluxerr_cols])

    z_valid = np.isfinite(z) & (z > 0) & (z < 15)
    print(f"    N={len(cat):,}  z_valid={z_valid.sum():,}  "
          f"z=[{z[z_valid].min():.2f},{z[z_valid].max():.2f}]  "
          f"logM=[{np.nanmin(logM):.1f},{np.nanmax(logM):.1f}]")
    return {"flux": flux, "fluxerr": fluxerr, "z": z, "logM": logM, "logSFR": logSFR}


def flux_to_mag(flux_ujy):
    """Convert flux in µJy to AB magnitude; returns 99.0 for non-detections."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(
            np.isfinite(flux_ujy) & (flux_ujy > 0),
            -2.5 * np.log10(flux_ujy / 3631e6),
            99.0,
        )
    return mag


def inject_noise_and_detect(sx, atlas_sed, atlas_z, min_det_bands):
    """Inject noise into atlas SEDs via sbipix API; return sigma_mag, flux_noisy, det_mask."""
    n_gal = atlas_sed.shape[0]

    # Set atlas SEDs as noiseless magnitudes in sbipix
    sx.obs = flux_to_mag(atlas_sed)          # (n_gal, n_filt) magnitude array
    sx.n_simulation = n_gal

    print(f"  Injecting noise...")
    sx.add_noise_nan_limit_all()             # fills sx.mag (n_gal, n_filt, 2)

    noisy_mag = sx.mag[:, :, 0]             # (n_gal, n_filt)
    sigma_mag  = sx.mag[:, :, 1]            # (n_gal, n_filt)

    # Convert noisy magnitude back to flux for plots
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_noisy = np.where(
            np.isfinite(noisy_mag) & (noisy_mag < 98.0),
            3631e6 * 10 ** (-0.4 * noisy_mag),
            0.0,
        )

    # Non-detections have noisy_mag = 99.0 (mag-space output)
    detected = (noisy_mag < 98.0) & np.isfinite(noisy_mag)
    n_det = np.sum(detected, axis=1)
    det_mask = n_det >= min_det_bands if min_det_bands > 0 else np.ones(n_gal, dtype=bool)

    return sigma_mag, flux_noisy, det_mask


def get_sigma_from_real(flux, fluxerr):
    """Compute σ_mag = (2.5/ln10) * fluxerr/flux for real detected galaxies."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (2.5 / np.log(10)) * np.abs(fluxerr / np.where(flux > 0, flux, np.nan))
    sigma[~np.isfinite(sigma)] = np.nan
    return sigma


# ── plots ──────────────────────────────────────────────────────────────────

def plot_sigma_vs_mag_single(filt_idx, filt_name, mock_sigma, mock_flux,
                              real_cats, outdir, tag=""):
    """σ_mag vs magnitude for one filter."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # real catalogs
    for key, cat in real_cats.items():
        if cat is None:
            continue
        f  = cat["flux"][:, filt_idx]
        fe = cat["fluxerr"][:, filt_idx]
        ok = (f > 0) & (fe > 0) & np.isfinite(f) & np.isfinite(fe)
        if ok.sum() < 5:
            continue
        mag_r = flux_ujy_to_mag(f[ok])
        sig_r = get_sigma_from_real(f[ok], fe[ok])
        ok2 = np.isfinite(mag_r) & np.isfinite(sig_r) & (sig_r > 0) & (sig_r < 5)
        ax.scatter(mag_r[ok2], sig_r[ok2], s=1, alpha=0.15,
                   color=CAT_COLORS[key], label=CAT_LABELS[key], rasterized=True)

    # mock atlas
    ok_m = (mock_flux[:, filt_idx] > 0) & np.isfinite(mock_sigma[:, filt_idx])
    if ok_m.sum() > 0:
        mag_m = flux_ujy_to_mag(mock_flux[ok_m, filt_idx])
        sig_m = mock_sigma[ok_m, filt_idx]
        ok2   = np.isfinite(mag_m) & np.isfinite(sig_m) & (sig_m > 0) & (sig_m < 5)
        ax.scatter(mag_m[ok2], sig_m[ok2], s=1, alpha=0.2,
                   color="k", label="Mock atlas", rasterized=True)
        # running median
        mbins = np.linspace(np.nanpercentile(mag_m[ok2], 1), np.nanpercentile(mag_m[ok2], 99), 30)
        mcen  = 0.5 * (mbins[:-1] + mbins[1:])
        med   = [np.nanmedian(sig_m[ok2][(mag_m[ok2] >= lo) & (mag_m[ok2] < hi)])
                 for lo, hi in zip(mbins[:-1], mbins[1:])]
        ax.plot(mcen, med, "k-", lw=2, label="Mock median")

    ax.set_xlabel(f"{filt_name} magnitude (AB)", fontsize=11)
    ax.set_ylabel(r"$\sigma_{\rm mag}$", fontsize=11)
    ax.set_xlim(17, 32)
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=8, markerscale=5)
    ax.set_title(f"{filt_name}{tag}")
    fig.tight_layout()
    fname = outdir / f"sigma_vs_mag_{filt_name.replace('-','_')}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_sigma_grid(mock_sigma, mock_flux, det_mask, real_cats, outdir):
    """2×5 grid of σ vs mag for all 10 bands."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    axes = axes.flatten()

    for j, (ax, fname) in enumerate(zip(axes, FILTER_SHORT)):
        # real catalogs
        for key, cat in real_cats.items():
            if cat is None:
                continue
            f  = cat["flux"][:, j]
            fe = cat["fluxerr"][:, j]
            ok = (f > 0) & (fe > 0) & np.isfinite(f) & np.isfinite(fe)
            if ok.sum() < 5:
                continue
            mag_r = flux_ujy_to_mag(f[ok])
            sig_r = get_sigma_from_real(f[ok], fe[ok])
            ok2   = np.isfinite(mag_r) & np.isfinite(sig_r) & (sig_r > 0) & (sig_r < 5)
            ax.scatter(mag_r[ok2], sig_r[ok2], s=0.5, alpha=0.1,
                       color=CAT_COLORS[key], rasterized=True)

        # mock atlas (all)
        ok_m = (mock_flux[:, j] > 0) & np.isfinite(mock_sigma[:, j])
        if ok_m.sum() > 0:
            mag_m = flux_ujy_to_mag(mock_flux[ok_m, j])
            sig_m = mock_sigma[ok_m, j]
            ok2   = np.isfinite(mag_m) & np.isfinite(sig_m) & (sig_m > 0) & (sig_m < 5)
            mbins = np.linspace(np.nanpercentile(mag_m[ok2], 1), np.nanpercentile(mag_m[ok2], 99), 25)
            mcen  = 0.5 * (mbins[:-1] + mbins[1:])
            med   = [np.nanmedian(sig_m[ok2][(mag_m[ok2] >= lo) & (mag_m[ok2] < hi)])
                     for lo, hi in zip(mbins[:-1], mbins[1:])]
            ax.plot(mcen, med, "k-", lw=1.5)

        ax.set_title(fname, fontsize=9)
        ax.set_xlim(17, 32)
        ax.set_ylim(0, 2.5)
        ax.tick_params(labelsize=7)

    for ax in axes[::5]:
        ax.set_ylabel(r"$\sigma_{\rm mag}$", fontsize=8)
    for ax in axes[5:]:
        ax.set_xlabel("mag (AB)", fontsize=7)

    # legend on last panel
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=CAT_COLORS[k], lw=2, label=CAT_LABELS[k])
               for k in CAT_COLORS if k in real_cats and real_cats[k] is not None]
    handles.append(Line2D([0],[0], color="k", lw=2, label="Mock median"))
    axes[-1].legend(handles=handles, fontsize=7)

    fig.suptitle("σ vs magnitude — all filters", fontsize=12)
    fig.tight_layout()
    out = outdir / "sigma_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_mag_grid(mock_flux, det_mask, real_cats, outdir):
    """2×5 grid of magnitude distributions (real vs mock all vs mock detected)."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes = axes.flatten()

    for j, (ax, fname) in enumerate(zip(axes, FILTER_SHORT)):
        bins = np.linspace(16, 32, 40)
        for key, cat in real_cats.items():
            if cat is None:
                continue
            f = cat["flux"][:, j]
            ok = (f > 0) & np.isfinite(f)
            if ok.sum() < 5:
                continue
            mag_r = flux_ujy_to_mag(f[ok])
            ok2   = np.isfinite(mag_r)
            ax.hist(mag_r[ok2], bins=bins, density=True, histtype="step",
                    color=CAT_COLORS[key], lw=1.2, label=CAT_LABELS[key])

        # mock all
        f_m = mock_flux[:, j]
        ok_m = (f_m > 0) & np.isfinite(f_m)
        if ok_m.sum() > 5:
            mag_m = flux_ujy_to_mag(f_m[ok_m])
            ok2   = np.isfinite(mag_m)
            ax.hist(mag_m[ok2], bins=bins, density=True, histtype="step",
                    color="k", lw=1.2, ls="--", label="Mock all")

        # mock detected
        ok_d = ok_m & det_mask
        if ok_d.sum() > 5:
            mag_d = flux_ujy_to_mag(f_m[ok_d])
            ok2   = np.isfinite(mag_d)
            ax.hist(mag_d[ok2], bins=bins, density=True, histtype="step",
                    color="gray", lw=1.2, ls=":", label="Mock det")

        ax.set_title(fname, fontsize=9)
        ax.set_xlim(16, 32)
        ax.tick_params(labelsize=7)

    for ax in axes[5:]:
        ax.set_xlabel("mag (AB)", fontsize=7)
    for ax in axes[::5]:
        ax.set_ylabel("density", fontsize=8)

    from matplotlib.lines import Line2D
    handles  = [Line2D([0],[0], color=CAT_COLORS[k], lw=2, label=CAT_LABELS[k])
                for k in CAT_COLORS if k in real_cats and real_cats[k] is not None]
    handles += [Line2D([0],[0], color="k", ls="--", lw=2, label="Mock all"),
                Line2D([0],[0], color="gray", ls=":", lw=2, label="Mock det")]
    axes[-1].legend(handles=handles, fontsize=7)

    fig.suptitle("Magnitude distributions — all filters", fontsize=12)
    fig.tight_layout()
    out = outdir / "mag_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_mag_vs_z(mock_flux, mock_z, det_mask, real_cats, outdir):
    """VIS magnitude vs redshift."""
    vis_idx = FILTER_SHORT.index("VIS")
    fig, ax = plt.subplots(figsize=(7, 5))

    for key, cat in real_cats.items():
        if cat is None:
            continue
        f = cat["flux"][:, vis_idx]
        z = cat["z"]
        ok = (f > 0) & np.isfinite(f) & np.isfinite(z) & (z > 0)
        mag_r = flux_ujy_to_mag(f[ok])
        ok2 = np.isfinite(mag_r)
        ax.scatter(z[ok][ok2], mag_r[ok2], s=1, alpha=0.1,
                   color=CAT_COLORS[key], label=CAT_LABELS[key], rasterized=True)

    # mock detected
    ok_d = det_mask & (mock_flux[:, vis_idx] > 0) & np.isfinite(mock_flux[:, vis_idx])
    if ok_d.sum() > 0:
        mag_d = flux_ujy_to_mag(mock_flux[ok_d, vis_idx])
        ok2   = np.isfinite(mag_d)
        ax.scatter(mock_z[ok_d][ok2], mag_d[ok2], s=1, alpha=0.15,
                   color="k", label="Mock detected", rasterized=True)

    ax.set_xlabel("Redshift", fontsize=11)
    ax.set_ylabel("VIS magnitude (AB)", fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(17, 32)
    ax.legend(fontsize=9, markerscale=5)
    ax.set_title("VIS mag vs redshift")
    fig.tight_layout()
    out = outdir / "mag_vs_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sigma_vs_z(mock_sigma, mock_z, det_mask, real_cats, outdir):
    """Per-filter median σ vs redshift (real + mock detected)."""
    z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    z_cen  = 0.5 * (z_bins[:-1] + z_bins[1:])

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
    axes = axes.flatten()

    for j, (ax, fname) in enumerate(zip(axes, FILTER_SHORT)):
        for key, cat in real_cats.items():
            if cat is None:
                continue
            f  = cat["flux"][:, j]
            fe = cat["fluxerr"][:, j]
            z  = cat["z"]
            ok = (f > 0) & (fe > 0) & np.isfinite(f) & np.isfinite(fe) & np.isfinite(z) & (z > 0)
            if ok.sum() < 10:
                continue
            sig_r = get_sigma_from_real(f[ok], fe[ok])
            med = [np.nanmedian(sig_r[(z[ok] >= lo) & (z[ok] < hi) & np.isfinite(sig_r)])
                   for lo, hi in zip(z_bins[:-1], z_bins[1:])]
            ax.plot(z_cen, med, "o-", ms=4, color=CAT_COLORS[key], label=CAT_LABELS[key])

        # mock detected
        ok_d = det_mask & np.isfinite(mock_z) & (mock_z > 0)
        if ok_d.sum() > 10:
            sig_m = mock_sigma[ok_d, j]
            z_d   = mock_z[ok_d]
            med_m = [np.nanmedian(sig_m[(z_d >= lo) & (z_d < hi) & np.isfinite(sig_m)])
                     for lo, hi in zip(z_bins[:-1], z_bins[1:])]
            ax.plot(z_cen, med_m, "s--", ms=4, color="k", label="Mock det")

        ax.set_title(fname, fontsize=9)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.5)
        ax.tick_params(labelsize=7)

    for ax in axes[5:]:
        ax.set_xlabel("Redshift", fontsize=7)
    for ax in axes[::5]:
        ax.set_ylabel(r"median $\sigma_{\rm mag}$", fontsize=8)

    from matplotlib.lines import Line2D
    handles  = [Line2D([0],[0], color=CAT_COLORS[k], lw=2, label=CAT_LABELS[k])
                for k in CAT_COLORS if k in real_cats and real_cats[k] is not None]
    handles.append(Line2D([0],[0], color="k", ls="--", lw=2, label="Mock detected"))
    axes[-1].legend(handles=handles, fontsize=7)

    fig.suptitle(r"Median $\sigma_{\rm mag}$ vs redshift", fontsize=12)
    fig.tight_layout()
    out = outdir / "sigma_vs_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_coverage(mock_z, det_mask, real_cats, outdir):
    """Detection fraction vs redshift for mock and real catalogs."""
    z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    z_cen  = 0.5 * (z_bins[:-1] + z_bins[1:])

    fig, ax = plt.subplots(figsize=(7, 5))

    # mock
    ok_z = np.isfinite(mock_z) & (mock_z > 0) & (mock_z < 5)
    frac_m = []
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        in_bin = ok_z & (mock_z >= lo) & (mock_z < hi)
        frac_m.append(det_mask[in_bin].mean() if in_bin.sum() > 0 else np.nan)
    ax.plot(z_cen, frac_m, "k-o", ms=5, label="Mock atlas")

    ax.set_xlabel("Redshift", fontsize=11)
    ax.set_ylabel("Detection fraction", fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f"Detection fraction vs z (SNR≥3, ≥{args_global.min_det_bands} bands)")
    fig.tight_layout()
    out = outdir / "coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_ssfr_vs_z(real_cats, mock_theta, det_mask, outdir):
    """sSFR vs redshift: real catalogs vs mock (all + detected)."""
    z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    z_cen  = 0.5 * (z_bins[:-1] + z_bins[1:])

    def _profile(ssfr, z):
        med = np.full(len(z_cen), np.nan)
        p16 = np.full(len(z_cen), np.nan)
        p84 = np.full(len(z_cen), np.nan)
        for i, (lo, hi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
            m = np.isfinite(ssfr) & np.isfinite(z) & (z >= lo) & (z < hi) & (ssfr > -15) & (ssfr < 0)
            if m.sum() >= 5:
                med[i], p16[i], p84[i] = np.percentile(ssfr[m], [50, 16, 84])
        return med, p16, p84

    def _ms_ssfr(z):
        # Schreiber+15 approximate main sequence sSFR
        coeff = np.where(z < 1, 1.0, np.where(z < 2, 2.0, 2.8))
        return -10.0 + coeff * np.log10(1 + z)

    fig, ax = plt.subplots(figsize=(8, 5))

    for key, cat in real_cats.items():
        if cat is None:
            continue
        z    = cat["z"]
        logM = cat["logM"]
        logSFR = cat["logSFR"]
        ok = (np.isfinite(z) & (z > 0) & np.isfinite(logM) & (logM > 5)
              & np.isfinite(logSFR) & (logSFR > -5))
        if ok.sum() < 10:
            continue
        ssfr = logSFR[ok] - logM[ok]
        med, p16, p84 = _profile(ssfr, z[ok])
        ax.plot(z_cen, med, "o-", ms=4, color=CAT_COLORS[key], label=CAT_LABELS[key])
        ax.fill_between(z_cen, p16, p84, alpha=0.15, color=CAT_COLORS[key])

    # mock all
    mstar = mock_theta["mstar"]
    sfr   = mock_theta["sfr"]
    z_m   = mock_theta["z"]
    ok_m  = np.isfinite(mstar) & (mstar > 5) & np.isfinite(sfr) & np.isfinite(z_m) & (z_m > 0)
    if ok_m.sum() > 10:
        ssfr_m = sfr[ok_m] - mstar[ok_m]
        med_m, p16_m, p84_m = _profile(ssfr_m, z_m[ok_m])
        ax.plot(z_cen, med_m, "k--", ms=4, lw=1.5, label="Mock all")
        ax.fill_between(z_cen, p16_m, p84_m, alpha=0.10, color="k")

    # mock detected
    ok_d = ok_m & det_mask
    if ok_d.sum() > 10:
        ssfr_d = sfr[ok_d] - mstar[ok_d]
        med_d, p16_d, p84_d = _profile(ssfr_d, z_m[ok_d])
        ax.plot(z_cen, med_d, "ks-", ms=4, lw=1.5, label="Mock detected")
        ax.fill_between(z_cen, p16_d, p84_d, alpha=0.10, color="gray")

    # Schreiber+15 main sequence reference
    z_ref = np.linspace(0.05, 5.0, 100)
    ax.plot(z_ref, _ms_ssfr(z_ref), "r--", lw=1.5, label="Schreiber+15 MS")

    ax.set_xlabel("Redshift", fontsize=11)
    ax.set_ylabel(r"$\log_{10}$(sSFR / yr$^{-1}$)", fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(-13, -7)
    ax.legend(fontsize=9)
    ax.set_title("sSFR vs redshift")
    fig.tight_layout()
    out = outdir / "ssfr_vs_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── main ──────────────────────────────────────────────────────────────────

args_global = None  # set in main() so plot functions can access args


def parse_args():
    p = argparse.ArgumentParser(description="Atlas noise/magnitude diagnostic")
    p.add_argument("--atlas", type=str,
                   default="atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas",
                   help="Atlas filename in library/ (default: atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas)")
    p.add_argument("--phot-type", type=str, default="templfit",
                   choices=["templfit", "2fwhm", "3fwhm"],
                   help="Photometry type for noise model and real catalogs (default: templfit)")
    p.add_argument("--catalogs", type=str, nargs="+",
                   default=["cosmos_deep", "cosmos_web", "desi"],
                   choices=["cosmos_deep", "cosmos_web", "desi"],
                   help="Real catalogs to overlay (default: all three)")
    p.add_argument("--min-det-bands", type=int, default=3,
                   help="Min bands with SNR≥3 to count as detected (0=no filter, default: 3)")
    p.add_argument("--outdir", type=str, default="sbi-logs/diagnose_v1.0",
                   help="Output directory (default: sbi-logs/diagnose_v1.0)")
    return p.parse_args()


def main():
    global args_global

    args = parse_args()
    args_global = args

    print("=" * 60)
    print("diagnose_sigma_vs_mag")
    print(f"  Atlas        : {args.atlas}")
    print(f"  Phot         : {args.phot_type}")
    print(f"  Catalogs     : {args.catalogs}")
    print(f"  min-det-bands: {args.min_det_bands}  (0=no filter)")
    print(f"  Outdir       : {args.outdir}")
    print("=" * 60)
    print()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load sbipix ──────────────────────────────────────────────────
    from sbipix import sbipix
    import hickle

    noise_prefix = f"north_{args.phot_type}"

    sx = sbipix()
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(OBS_DIR),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    sx.configure_noise_model(
        sigma_sampler="mag_lognormal",
        detection_model="hard",
        observation_space="mag",
    )
    sx.load_obs_features()
    print(f"Sigma samples loaded: {sx.sigma_samples_obs is not None}")
    print("Observational features loaded")

    # ── load atlas ───────────────────────────────────────────────────
    atlas_path = LIB_DIR / args.atlas
    print(f"\nAtlas: {args.atlas}")
    data = hickle.load(str(atlas_path))
    # keys are top-level for this atlas format
    key_prefix = "data/" if "data/mstar" in data else ""
    mstar = np.array(data[key_prefix + "mstar"], dtype=float)
    sfr   = np.array(data[key_prefix + "sfr"],   dtype=float)
    zval  = np.array(data[key_prefix + "zval"],  dtype=float)
    sed   = np.array(data[key_prefix + "sed"],   dtype=float)   # (n_gal, n_filt) in µJy

    n_total = len(mstar)
    print(f"  Loaded {n_total} galaxies from hickle")

    # physical mask: logM > 5
    phys = mstar > 5
    mstar = mstar[phys]
    sfr   = sfr[phys]
    zval  = zval[phys]
    sed   = sed[phys]
    print(f"  Physical mask: {phys.sum()} / {n_total} pass (logM>5)")
    print(f"  logM=[{mstar.min():.1f},{mstar.max():.1f}]  z=[{zval.min():.2f},{zval.max():.2f}]")

    # ── load real catalogs ───────────────────────────────────────────
    print("\nLoading obs catalogs...")
    real_cats = {}
    for key in args.catalogs:
        cat = load_catalog(key, args.phot_type)
        if cat is not None:
            real_cats[key] = cat

    # ── inject noise ─────────────────────────────────────────────────
    sigma_mag, flux_noisy, det_mask = inject_noise_and_detect(sx, sed, zval, args.min_det_bands)

    # summary
    n_pass = det_mask.sum()
    print(f"\n  Detection filter: SNR≥3.0 in ≥{args.min_det_bands} bands")
    print(f"  Pass: {n_pass}/{len(det_mask)} ({100*n_pass/len(det_mask):.1f}%)")
    z_edges = [0, 1, 2, 3, 4, 5]
    for lo, hi in zip(z_edges[:-1], z_edges[1:]):
        in_bin = (zval >= lo) & (zval < hi)
        n_bin  = in_bin.sum()
        n_det  = (det_mask & in_bin).sum()
        frac   = 100 * n_det / n_bin if n_bin > 0 else 0
        print(f"    z=[{lo},{hi}): {n_det}/{n_bin} ({frac:.1f}%)")

    # per-filter sigma stats (after detection)
    print(f"\n  σ-debug (after det-filter, {n_pass} galaxies):")
    for j, fname in enumerate(FILTER_SHORT):
        sm = sigma_mag[det_mask, j]
        ok = np.isfinite(sm) & (sm > 0)
        p50 = np.nanpercentile(sm[ok], 50) if ok.sum() > 0 else np.nan
        print(f"    [{j}] {fname:<12} p50_sigma={p50:.3f}  in_domain={ok.sum()}/{n_pass}")

    # ── plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # VIS and NISP-H individual panels
    vis_idx = FILTER_SHORT.index("VIS")
    h_idx   = FILTER_SHORT.index("NISP-H")
    plot_sigma_vs_mag_single(vis_idx, "VIS", sigma_mag, flux_noisy, real_cats, outdir)
    plot_sigma_vs_mag_single(h_idx,   "NISP-H", sigma_mag, flux_noisy, real_cats, outdir)

    plot_sigma_vs_z(sigma_mag, zval, det_mask, real_cats, outdir)
    plot_coverage(zval, det_mask, real_cats, outdir)
    plot_sigma_grid(sigma_mag, flux_noisy, det_mask, real_cats, outdir)
    plot_mag_vs_z(flux_noisy, zval, det_mask, real_cats, outdir)
    plot_mag_grid(flux_noisy, det_mask, real_cats, outdir)

    mock_theta = {"mstar": mstar, "sfr": sfr, "z": zval}
    plot_ssfr_vs_z(real_cats, mock_theta, det_mask, outdir)

    print(f"\nDone. All plots → {args.outdir}/")


if __name__ == "__main__":
    main()
