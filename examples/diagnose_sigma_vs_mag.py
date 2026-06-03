"""
diagnose_sigma_vs_mag.py — sigma_mag vs magnitude: real surveys vs mock atlas.

For NISP-H and VIS, plots sigma_mag vs observed magnitude side-by-side
(real surveys | mock atlas), color-coded by z, logM, logSFR, sSFR.

Supported obs catalogs: cosmos_deep, cosmos_web, desi (any combination).
Supports any atlas file via direct hickle loading (no dense_basis naming rules).

Usage:
    python examples/diagnose_sigma_vs_mag.py \\
        --atlas-name atlas_100k_SFRflat.dbatlas \\
        --phot-type templfit \\
        --catalogs cosmos_deep cosmos_web desi \\
        --outdir sbi-logs/diagnose_sigma_SFRflat \\
        2>&1 | tee sbi-logs/diagnose_sigma_SFRflat.log

    python examples/diagnose_sigma_vs_mag.py \\
        --atlas-name atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas \\
        --outdir sbi-logs/diagnose_sigma_100k \\
        2>&1 | tee sbi-logs/diagnose_sigma_100k.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbipix.utils.sed_utils import load_filter_metadata, flux_ujy_to_mag

_OBS_DIR = Path(__file__).resolve().parents[1] / "obs" / "obs_properties"
_LIB_DIR = Path(__file__).resolve().parents[1] / "library"

_FILTER_META     = load_filter_metadata("filters_to_use.dat", filt_dir=str(_OBS_DIR))
FILTER_SHORT     = [m["short"]    for m in _FILTER_META]
FILTER_COL_STEMS = [m["col_stem"] for m in _FILTER_META]

LN10       = np.log(10.0)
NONDET_MAG = 99.0
SNR_MIN    = 3.0

PLOT_FILTERS = [
    ("VIS",    FILTER_SHORT.index("VIS")),
    ("NISP-H", FILTER_SHORT.index("NISP-H")),
]

PARAM_CONFIGS = [
    ("z",      r"Redshift $z$",                  (0.0,  4.5),  "plasma"),
    ("logM",   r"$\log M_*$",                    (6.5, 11.5),  "viridis"),
    ("logSFR", r"$\log \mathrm{SFR}$",           (-3.5, 2.5),  "RdYlBu_r"),
    ("sSFR",   r"$\log \mathrm{sSFR}$",          (-13., -8.5), "coolwarm"),
]

# Catalog color/style for sigma_vs_z overlay
CAT_STYLES = {
    "cosmos_deep": dict(color="steelblue",  ls="-",  lw=2),
    "cosmos_web":  dict(color="darkorange", ls="--", lw=2),
    "desi":        dict(color="forestgreen",ls=":",  lw=2),
    "mock":        dict(color="tomato",     ls="-",  lw=2),
}

CATALOG_SPECS = {
    "cosmos_deep": {
        "path": str(_OBS_DIR / "COSMOS_DEEP_PHZ.fits"),
        "z_col":     "PHZ_PP_MEDIAN_REDSHIFT",
        "logM_col":  "PHZ_PP_MEDIAN_STELLARMASS",   # already log10
        "logSFR_col":"PHZ_PP_MEDIAN_SFR",            # already log10
        "logM_is_log": True,
        "logSFR_is_log": True,
        "z_valid": lambda z: np.isfinite(z) & (z > 0),
        "label": "COSMOS-Deep",
    },
    "cosmos_web": {
        "path": str(_OBS_DIR / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"),
        "z_col":     "z_lephare",
        "logM_col":  "logM_lephare",                 # already log10
        "logSFR_col":"logSFR_lephare",               # already log10
        "logM_is_log": True,
        "logSFR_is_log": True,
        "z_valid": lambda z: np.isfinite(z) & (z > 0) & (z < 12),
        "label": "COSMOS-Web",
    },
    "desi": {
        "path": str(_OBS_DIR / "COSMOS-DESI" / "matched_euclid_desi.fits"),
        "z_col":     "z_desi",
        "logM_col":  "logM_desi_Cigale",             # already log10
        "logSFR_col":"logSFR_desi_Cigale",           # already log10
        "logM_is_log": True,
        "logSFR_is_log": True,
        "z_valid": lambda z: np.isfinite(z) & (z > 0),
        "label": "DESI",
    },
}


# ---------------------------------------------------------------------------
# Load obs catalogs
# ---------------------------------------------------------------------------

def load_catalog(name, phot_type, snr_min=SNR_MIN):
    """Load one obs catalog. Returns dict with mag/sigma/z/logM/logSFR/sSFR."""
    from astropy.io import fits
    spec = CATALOG_SPECS[name]
    print(f"  [{name}] Loading: {Path(spec['path']).name}")
    hdul = fits.open(spec["path"])
    cat  = hdul[1].data
    N    = len(cat)

    n_filt = len(FILTER_SHORT)
    mag    = np.full((n_filt, N), np.nan)
    sigma  = np.full((n_filt, N), np.nan)

    for fi, (stem, short) in enumerate(zip(FILTER_COL_STEMS, FILTER_SHORT)):
        if phot_type == "templfit":
            fcol = "flux_vis_psf"   if short == "VIS" else f"flux_{stem}_templfit"
            ecol = "fluxerr_vis_psf" if short == "VIS" else f"fluxerr_{stem}_templfit"
        else:
            fcol = f"flux_{stem}_{phot_type}_aper"
            ecol = f"fluxerr_{stem}_{phot_type}_aper"
        if fcol not in cat.names:
            continue
        flux = np.asarray(cat[fcol], float)
        err  = np.asarray(cat[ecol], float) if ecol in cat.names else np.full(N, np.nan)
        det  = np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux > 0) & (flux / err >= snr_min)
        mag[fi, det]   = flux_ujy_to_mag(flux[det])
        sigma[fi, det] = (2.5 / LN10) * err[det] / flux[det]

    z_raw = np.asarray(cat[spec["z_col"]],     float)
    m_raw = np.asarray(cat[spec["logM_col"]],  float)
    s_raw = np.asarray(cat[spec["logSFR_col"]], float)

    z_ok = spec["z_valid"](z_raw)
    z    = np.where(z_ok, z_raw, np.nan)
    logM = np.where(np.isfinite(m_raw) & (m_raw > 0), m_raw, np.nan)
    logSFR = np.where(np.isfinite(s_raw), s_raw, np.nan)
    sSFR   = logSFR - logM

    hdul.close()
    n_valid = np.isfinite(z).sum()
    print(f"    N={N:,}  z_valid={n_valid:,}  "
          f"z=[{np.nanmin(z):.2f},{np.nanmax(z):.2f}]  "
          f"logM=[{np.nanmin(logM):.1f},{np.nanmax(logM):.1f}]")
    return {"mag": mag, "sigma": sigma, "z": z, "logM": logM,
            "logSFR": logSFR, "sSFR": sSFR, "label": spec["label"]}


# ---------------------------------------------------------------------------
# Load atlas + inject noise
# ---------------------------------------------------------------------------

def load_atlas_and_inject_noise(args):
    """
    Load atlas via hickle (handles any atlas name) and run sbipix noise pipeline.
    Returns (theta_dict, obs_mag_true, mock_mag, mock_sigma, model).
    """
    import hickle
    from sbipix import sbipix

    atlas_path = _LIB_DIR / args.atlas_name
    if not atlas_path.exists():
        # try without any suffix transformation
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    print(f"\nAtlas: {args.atlas_name}")
    d = hickle.load(str(atlas_path))
    n_raw = len(d["mstar"])
    print(f"  Loaded {n_raw} galaxies from hickle")

    # Detect log10 vs linear for mstar / sfr
    mstar = np.asarray(d["mstar"], float)
    sfr   = np.asarray(d["sfr"],   float)

    fin = mstar[np.isfinite(mstar)]
    logM = mstar if (fin.size > 0 and np.nanpercentile(fin, 99) < 100) else np.log10(np.clip(mstar, 1e-300, None))

    fin_s = sfr[np.isfinite(sfr)]
    if fin_s.size > 0 and np.nanpercentile(fin_s[fin_s > -50], 99) < 100:
        logSFR = sfr
    else:
        logSFR = np.log10(np.clip(sfr, 1e-300, None))

    z    = np.asarray(d["zval"], float)
    sSFR = logSFR - logM

    # Physical mask: logM > 5
    ok = np.isfinite(logM) & (logM > 5) & (logM < 13) & np.isfinite(logSFR) & np.isfinite(z)
    print(f"  Physical mask: {ok.sum()} / {n_raw} pass (logM>5)")

    logM   = logM[ok];   logSFR = logSFR[ok];  z = z[ok];  sSFR = sSFR[ok]
    sed    = np.asarray(d["sed"], float)[ok]   # (n, n_filt) µJy

    # Convert µJy SED to AB magnitudes
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_mag_true = np.where(sed > 0, -2.5 * np.log10(sed * 1e-6 / 3631.0), NONDET_MAG)

    n_gal = ok.sum()
    noise_prefix = f"north_{args.phot_type}"

    model = sbipix()
    model.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(_OBS_DIR),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    model.sigma_samples_file = f"sigma_samples_{noise_prefix}.npy"
    model.atlas_path  = str(_LIB_DIR) + "/"
    model.model_path  = str(_LIB_DIR) + "/"
    model.parametric  = True
    model.both_masses = True
    model.infer_z     = False
    model.include_limit  = True
    model.include_sigma  = True
    model.configure_noise_model(
        sigma_sampler="mag_lognormal",
        detection_model="hard",
        observation_space="flux",
    )
    model.snr_threshold = SNR_MIN

    # Inject atlas directly into model
    model.obs           = obs_mag_true        # (n_gal, n_filt)
    model.n_simulation  = n_gal

    print(f"  Loading noise features ({noise_prefix})...")
    model.load_obs_features()

    print("  Injecting noise...")
    model.add_noise_nan_limit_all()

    # Extract noisy mag + sigma
    obs_space = getattr(model, "noise_observation_space", "mag")
    first  = model.mag[:, :, 0].T   # (n_filt, n_gal)
    second = model.mag[:, :, 1].T

    if obs_space == "flux":
        mf = first
        sf = np.maximum(second, 1e-12)
        mock_mag   = np.where(mf > 0, flux_ujy_to_mag(mf), NONDET_MAG)
        mock_sigma = np.where(mf > 0, (2.5 / LN10) * sf / np.maximum(np.abs(mf), 1e-12), np.nan)
    else:
        mock_mag   = first
        mock_sigma = second

    theta_dict = {"logM": logM, "logSFR": logSFR, "z": z, "sSFR": sSFR}
    print(f"  logM=[{logM.min():.1f},{logM.max():.1f}]  "
          f"z=[{z.min():.2f},{z.max():.2f}]")

    # σ-debug summary (before any detection filter)
    print(f"\n  σ-debug (all {n_gal} galaxies, no detection filter):")
    for fi, band in enumerate(FILTER_SHORT):
        s = mock_sigma[fi]
        in_dom = np.isfinite(s) & (s > 0) & (s < 5)
        p50 = np.nanpercentile(s[np.isfinite(s)], 50) if np.isfinite(s).sum() > 0 else np.nan
        print(f"    [{fi}] {band:10s}  p50_sigma={p50:.3f}  in_domain={in_dom.sum():>6d}/{n_gal}")

    # Per-galaxy SNR: (2.5/ln10) / sigma_mag; detected band = SNR>=snr_threshold
    snr_per_band = np.where(
        np.isfinite(mock_sigma) & (mock_sigma > 0),
        (2.5 / LN10) / mock_sigma,
        0.0
    )
    n_det_per_gal = np.sum(
        (snr_per_band >= SNR_MIN) & (mock_mag < NONDET_MAG - 0.5), axis=0
    )   # (n_gal,)

    # Detection filter mask (unfiltered = all True if min_det_bands=0)
    min_det = getattr(args, "min_det_bands", 0)
    det_mask = n_det_per_gal >= min_det   # always True when min_det=0
    n_pass = det_mask.sum()

    # z-bin breakdown of detection filter
    print(f"\n  Detection filter: SNR≥{SNR_MIN} in ≥{min_det} bands")
    print(f"  Pass: {n_pass}/{n_gal} ({100*n_pass/n_gal:.1f}%)")
    for zlo, zhi in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
        in_z = (z >= zlo) & (z < zhi)
        n_z  = in_z.sum()
        n_ok = (in_z & det_mask).sum()
        if n_z > 0:
            print(f"    z=[{zlo},{zhi}): {n_ok}/{n_z} ({100*n_ok/n_z:.1f}%)")

    if min_det > 0:
        print(f"\n  σ-debug (after det-filter, {n_pass} galaxies):")
        for fi, band in enumerate(FILTER_SHORT):
            s = mock_sigma[fi][det_mask]
            in_dom = np.isfinite(s) & (s > 0) & (s < 5)
            p50 = np.nanpercentile(s[np.isfinite(s)], 50) if np.isfinite(s).sum() > 0 else np.nan
            print(f"    [{fi}] {band:10s}  p50_sigma={p50:.3f}  in_domain={in_dom.sum():>6d}/{n_pass}")

    return theta_dict, obs_mag_true.T, mock_mag, mock_sigma, model, det_mask


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scatter_panel(ax, mag, sigma, color_vals, vmin, vmax, cmap,
                   lut_edges=None, label=""):
    ok = (np.isfinite(sigma) & (sigma > 0) & (sigma < 10)
          & np.isfinite(mag)  & (mag < NONDET_MAG - 0.5)
          & np.isfinite(color_vals))
    if ok.sum() == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center"); return None
    x, y, c = mag[ok], sigma[ok], color_vals[ok]
    order = np.argsort(c)
    sc = ax.scatter(x[order], y[order], c=c[order], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=1, alpha=0.4, rasterized=True, linewidths=0)
    if lut_edges is not None:
        for e in lut_edges: ax.axvline(e, color="0.6", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlim(17, 30); ax.set_ylim(0.01, 10); ax.set_yscale("log")
    ax.set_xlabel("AB magnitude"); ax.set_ylabel(r"$\sigma_{\rm mag}$")
    ax.set_title(f"{label}  (N={ok.sum():,})", fontsize=8)
    return sc


def plot_sigma_vs_mag(primary_cat, mock_params, mock_mag, mock_sigma, model, outdir):
    """sigma vs mag: primary obs catalog (left) | mock (right), one figure per band."""
    perc = model.percentiles
    for band_name, fi in PLOT_FILTERS:
        lut_edges = None
        if fi < perc.shape[1]:
            e = perc[:, fi]; lut_edges = e[np.isfinite(e)]

        fig, axes = plt.subplots(len(PARAM_CONFIGS), 2,
                                  figsize=(10, 4 * len(PARAM_CONFIGS)),
                                  sharex=True, sharey=True)
        fig.suptitle(f"{band_name}  —  $\\sigma_{{\\rm mag}}$ vs magnitude\n"
                     f"Left: {primary_cat['label']}    Right: mock atlas", fontsize=11)

        for row, (pname, plabel, (vmin, vmax), cmap) in enumerate(PARAM_CONFIGS):
            sc_r = _scatter_panel(axes[row, 0], primary_cat["mag"][fi],
                                   primary_cat["sigma"][fi], primary_cat[pname],
                                   vmin, vmax, cmap, lut_edges,
                                   label=f"{primary_cat['label']} · {plabel}")
            sc_m = _scatter_panel(axes[row, 1], mock_mag[fi], mock_sigma[fi],
                                   mock_params[pname], vmin, vmax, cmap, lut_edges,
                                   label=f"mock · {plabel}")
            sc = sc_r if sc_r is not None else sc_m
            if sc is not None:
                fig.colorbar(sc, ax=list(axes[row]), fraction=0.03, pad=0.01, label=plabel)

        fig.tight_layout()
        out = outdir / f"sigma_vs_mag_{band_name.replace('-','_')}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Saved: {out.name}")


def _median_profile(mag, sigma, z_arr, z_bins):
    z_cen = 0.5 * (z_bins[:-1] + z_bins[1:])
    med = np.full(len(z_cen), np.nan)
    p16 = np.full(len(z_cen), np.nan)
    p84 = np.full(len(z_cen), np.nan)
    for i, (zlo, zhi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        ok = (np.isfinite(sigma) & (sigma > 0) & (sigma < 10)
              & (mag < NONDET_MAG - 0.5)
              & np.isfinite(z_arr) & (z_arr >= zlo) & (z_arr < zhi))
        if ok.sum() >= 5:
            med[i], p16[i], p84[i] = np.percentile(sigma[ok], [50, 16, 84])
    return med, p16, p84, z_cen


def _median_profile_mag(mag_arr, z_arr, z_bins):
    """Median + p16/p84 of AB mag per z bin (detected sources only)."""
    z_cen = 0.5 * (z_bins[:-1] + z_bins[1:])
    med = np.full(len(z_cen), np.nan)
    p16 = np.full(len(z_cen), np.nan)
    p84 = np.full(len(z_cen), np.nan)
    for i, (zlo, zhi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
        ok = (np.isfinite(mag_arr) & (mag_arr < NONDET_MAG - 0.5)
              & np.isfinite(z_arr) & (z_arr >= zlo) & (z_arr < zhi))
        if ok.sum() >= 5:
            med[i], p16[i], p84[i] = np.percentile(mag_arr[ok], [50, 16, 84])
    return med, p16, p84, z_cen


def plot_sigma_vs_z(all_cats, mock_params, mock_mag, mock_sigma, det_mask, min_det, outdir):
    """Median sigma_mag vs z: all obs catalogs + mock (unfiltered & filtered) per band."""
    z_bins = np.linspace(0.0, 5.0, 21)

    fig, axes = plt.subplots(1, len(PLOT_FILTERS), figsize=(6 * len(PLOT_FILTERS), 5))
    fig.suptitle(r"Median $\sigma_{\rm mag}$ vs redshift", fontsize=12)

    for ax, (band_name, fi) in zip(axes, PLOT_FILTERS):
        # real catalogs
        for cat in all_cats:
            med, p16, p84, z_cen = _median_profile(
                cat["mag"][fi], cat["sigma"][fi], cat["z"], z_bins)
            st = CAT_STYLES.get(cat["_key"], {})
            ax.fill_between(z_cen, p16, p84, alpha=0.15, color=st.get("color","gray"))
            ax.plot(z_cen, med, label=cat["label"], **st)

        # mock unfiltered
        med_m, p16_m, p84_m, z_cen = _median_profile(
            mock_mag[fi], mock_sigma[fi], mock_params["z"], z_bins)
        ax.fill_between(z_cen, p16_m, p84_m, alpha=0.10, color="tomato")
        ax.plot(z_cen, med_m, color="tomato", lw=1.5, ls="--", label="mock (all)")

        # mock filtered (if filter active)
        if det_mask is not None and min_det > 0:
            med_f, p16_f, p84_f, _ = _median_profile(
                mock_mag[fi][det_mask], mock_sigma[fi][det_mask],
                mock_params["z"][det_mask], z_bins)
            ax.fill_between(z_cen, p16_f, p84_f, alpha=0.15, color="crimson")
            ax.plot(z_cen, med_f, color="crimson", lw=2.5, ls="-",
                    label=f"mock (≥{min_det} det bands)")

        ax.set_xlabel("Redshift z"); ax.set_ylabel(r"$\sigma_{\rm mag}$")
        ax.set_yscale("log"); ax.set_title(band_name)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "sigma_vs_z.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_coverage(all_cats, mock_params, outdir):
    """(z, logM) and (z, sSFR) 2D histograms: one column per catalog + mock."""
    cats = all_cats + [{"z": mock_params["z"], "logM": mock_params["logM"],
                        "sSFR": mock_params["sSFR"], "label": "mock"}]
    n_cols = len(cats)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
    fig.suptitle("Parameter coverage", fontsize=12)

    kw = dict(bins=[40, 40], cmap="viridis", norm=mcolors.LogNorm(vmin=1))
    for ci, cat in enumerate(cats):
        ax = axes[0, ci]
        ok = np.isfinite(cat["z"]) & np.isfinite(cat["logM"])
        if ok.sum() > 1:
            h, _, _, img = ax.hist2d(cat["z"][ok], cat["logM"][ok],
                                      range=[[0,5],[5,12]], **kw)
            plt.colorbar(img, ax=ax, label="N")
        ax.set_title(cat["label"]); ax.set_xlabel("z"); ax.set_ylabel(r"$\log M_*$")

        ax = axes[1, ci]
        ok2 = np.isfinite(cat["z"]) & np.isfinite(cat["sSFR"])
        if ok2.sum() > 1:
            h, _, _, img2 = ax.hist2d(cat["z"][ok2], cat["sSFR"][ok2],
                                       range=[[0,5],[-14,-7]], **kw)
            plt.colorbar(img2, ax=ax, label="N")
        ax.set_xlabel("z"); ax.set_ylabel(r"$\log\,\mathrm{sSFR}$")

    fig.tight_layout()
    out = outdir / "coverage.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sigma_grid(primary_cat, mock_params, mock_mag, mock_sigma, det_mask, min_det, outdir):
    """2D median sigma in (z, logM) cells: obs catalog | mock-all | mock-filtered."""
    z_bins    = np.linspace(0.0, 5.0, 11)
    logM_bins = np.linspace(5.0, 12.0, 15)
    norm_s = mcolors.LogNorm(vmin=0.05, vmax=5.0)
    norm_n = mcolors.LogNorm(vmin=1)

    n_cols  = 6 if (det_mask is not None and min_det > 0) else 4
    n_bands = len(PLOT_FILTERS)
    fig, axes = plt.subplots(n_bands, n_cols, figsize=(4.5 * n_cols, 4.5 * n_bands))
    if n_bands == 1:
        axes = axes[np.newaxis, :]
    title_suffix = f"  |  filtered: ≥{min_det} det bands" if min_det > 0 else ""
    fig.suptitle(r"Median $\sigma_{\rm mag}$ in $(z, \log M_*)$ cells — "
                 f"{primary_cat['label']} | mock-all | mock-filtered{title_suffix}", fontsize=11)

    def _grid(mag_f, sigma_f, z, logM):
        z_cen = 0.5*(z_bins[:-1]+z_bins[1:]); m_cen = 0.5*(logM_bins[:-1]+logM_bins[1:])
        gmed = np.full((len(z_cen), len(m_cen)), np.nan)
        gn   = np.zeros_like(gmed, dtype=int)
        base = (np.isfinite(sigma_f) & (sigma_f>0) & (sigma_f<10)
                & (mag_f < NONDET_MAG-0.5) & np.isfinite(z) & np.isfinite(logM))
        for iz,(zlo,zhi) in enumerate(zip(z_bins[:-1],z_bins[1:])):
            for im,(mlo,mhi) in enumerate(zip(logM_bins[:-1],logM_bins[1:])):
                ok = base & (z>=zlo)&(z<zhi)&(logM>=mlo)&(logM<mhi)
                gn[iz,im] = ok.sum()
                if ok.sum()>=3: gmed[iz,im] = np.median(sigma_f[ok])
        return gmed, gn

    for row, (band_name, fi) in enumerate(PLOT_FILTERS):
        gmed_r, gn_r = _grid(primary_cat["mag"][fi], primary_cat["sigma"][fi],
                              primary_cat["z"], primary_cat["logM"])
        gmed_m, gn_m = _grid(mock_mag[fi], mock_sigma[fi],
                              mock_params["z"], mock_params["logM"])

        panels = [
            (gmed_r, gn_r, f"{band_name} {primary_cat['label']}: median σ", True),
            (gn_r,   gn_r, f"{band_name} {primary_cat['label']}: N",        False),
            (gmed_m, gn_m, f"{band_name} mock (all): median σ",             True),
            (gn_m,   gn_m, f"{band_name} mock (all): N",                    False),
        ]

        if det_mask is not None and min_det > 0:
            gmed_f, gn_f = _grid(mock_mag[fi][det_mask], mock_sigma[fi][det_mask],
                                  mock_params["z"][det_mask], mock_params["logM"][det_mask])
            panels += [
                (gmed_f, gn_f, f"{band_name} mock (≥{min_det} det): median σ", True),
                (gn_f,   gn_f, f"{band_name} mock (≥{min_det} det): N",        False),
            ]

        for col, (gmed, gn, title, is_sigma) in enumerate(panels):
            ax = axes[row, col]
            norm = norm_s if is_sigma else norm_n
            data = gmed if is_sigma else gn.astype(float)
            data_plot = np.where(data > 0, data, np.nan)
            im = ax.pcolormesh(z_bins, logM_bins, data_plot.T,
                               cmap="RdYlGn_r" if is_sigma else "viridis", norm=norm)
            plt.colorbar(im, ax=ax, label=r"median $\sigma$" if is_sigma else "N")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("z"); ax.set_ylabel(r"$\log M_*$")

    fig.tight_layout()
    out = outdir / "sigma_grid.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_mag_vs_z(all_cats, mock_params, obs_mag_true, mock_mag, det_mask, min_det, outdir):
    """Median AB magnitude vs z: obs + mock noiseless + mock noisy + mock filtered.

    Key diagnostic: if mock noiseless/noisy lines sit *above* (fainter AB mag)
    the real-catalog line at the same z, the SED model is too faint → SED issue.
    If they track real but sigma is still high → noise model issue.
    """
    z_bins = np.linspace(0.0, 5.0, 21)

    fig, axes = plt.subplots(1, len(PLOT_FILTERS), figsize=(6 * len(PLOT_FILTERS), 5))
    fig.suptitle(
        "Median AB magnitude vs redshift\n"
        "mock above real → SED too faint (SED issue); "
        "mock ≈ real but σ high → noise model issue",
        fontsize=10,
    )

    for ax, (band_name, fi) in zip(axes, PLOT_FILTERS):
        # real catalogs
        for cat in all_cats:
            med, p16, p84, z_cen = _median_profile_mag(cat["mag"][fi], cat["z"], z_bins)
            st = CAT_STYLES.get(cat["_key"], {})
            ax.fill_between(z_cen, p16, p84, alpha=0.15, color=st.get("color", "gray"))
            ax.plot(z_cen, med, label=cat["label"], **st)

        # mock noiseless (true SED magnitudes)
        med_t, p16_t, p84_t, z_cen = _median_profile_mag(obs_mag_true[fi], mock_params["z"], z_bins)
        ax.fill_between(z_cen, p16_t, p84_t, alpha=0.10, color="plum")
        ax.plot(z_cen, med_t, color="purple", lw=1.5, ls=":", label="mock (noiseless SED)")

        # mock noisy all
        med_m, p16_m, p84_m, _ = _median_profile_mag(mock_mag[fi], mock_params["z"], z_bins)
        ax.fill_between(z_cen, p16_m, p84_m, alpha=0.10, color="tomato")
        ax.plot(z_cen, med_m, color="tomato", lw=1.5, ls="--", label="mock noisy (all)")

        # mock filtered
        if det_mask is not None and min_det > 0:
            med_f, p16_f, p84_f, _ = _median_profile_mag(
                mock_mag[fi][det_mask], mock_params["z"][det_mask], z_bins)
            ax.fill_between(z_cen, p16_f, p84_f, alpha=0.15, color="crimson")
            ax.plot(z_cen, med_f, color="crimson", lw=2.5, ls="-",
                    label=f"mock (≥{min_det} det bands)")

        ax.invert_yaxis()   # brighter (lower AB mag) at top
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("AB magnitude (median, lower = brighter)")
        ax.set_title(band_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "mag_vs_z.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_mag_grid(primary_cat, mock_params, obs_mag_true, mock_mag, det_mask, min_det, outdir):
    """2D median AB magnitude in (z, logM) cells + Δmag = mock − real.

    Δmag > 0  →  mock is *fainter* than real at same (z, logM)  →  SED issue.
    Δmag ≈ 0 but σ_mock > σ_real                                →  noise model issue.
    """
    z_bins    = np.linspace(0.0, 5.0, 11)
    logM_bins = np.linspace(5.0, 12.0, 15)

    def _grid_mag(mag_f, z, logM):
        gmed = np.full((len(z_bins)-1, len(logM_bins)-1), np.nan)
        base = (np.isfinite(mag_f) & (mag_f < NONDET_MAG - 0.5)
                & np.isfinite(z) & np.isfinite(logM))
        for iz, (zlo, zhi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
            for im, (mlo, mhi) in enumerate(zip(logM_bins[:-1], logM_bins[1:])):
                ok = base & (z >= zlo) & (z < zhi) & (logM >= mlo) & (logM < mhi)
                if ok.sum() >= 3:
                    gmed[iz, im] = np.median(mag_f[ok])
        return gmed

    n_bands  = len(PLOT_FILTERS)
    has_filt = det_mask is not None and min_det > 0
    # columns: obs | mock-noiseless | mock-noisy | Δ(noisy−obs) [| mock-filtered | Δ(filt−obs)]
    n_cols = 6 if has_filt else 4

    fig, axes = plt.subplots(n_bands, n_cols, figsize=(4.5 * n_cols, 4.5 * n_bands))
    if n_bands == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        r"Median AB magnitude in $(z,\,\log M_*)$ cells — "
        f"{primary_cat['label']} | mock noiseless | mock noisy | Δ(mock−real)\n"
        r"$\Delta>0$ = mock fainter → SED issue;  $\Delta\approx0$ + high $\sigma$ → noise model issue",
        fontsize=10,
    )

    mag_norm   = mcolors.Normalize(vmin=18, vmax=28)
    delta_norm = mcolors.Normalize(vmin=-3, vmax=3)

    for row, (band_name, fi) in enumerate(PLOT_FILTERS):
        gmed_r = _grid_mag(primary_cat["mag"][fi], primary_cat["z"], primary_cat["logM"])
        gmed_t = _grid_mag(obs_mag_true[fi],        mock_params["z"], mock_params["logM"])
        gmed_m = _grid_mag(mock_mag[fi],             mock_params["z"], mock_params["logM"])
        delta_m = np.where(np.isfinite(gmed_m) & np.isfinite(gmed_r), gmed_m - gmed_r, np.nan)

        panels = [
            (gmed_r,  "RdYlBu_r", mag_norm,   f"{band_name} {primary_cat['label']} (real)"),
            (gmed_t,  "RdYlBu_r", mag_norm,   f"{band_name} mock noiseless SED"),
            (gmed_m,  "RdYlBu_r", mag_norm,   f"{band_name} mock noisy"),
            (delta_m, "RdBu_r",   delta_norm, f"{band_name} Δmag noisy − {primary_cat['label']}"),
        ]
        if has_filt:
            gmed_f  = _grid_mag(mock_mag[fi][det_mask], mock_params["z"][det_mask], mock_params["logM"][det_mask])
            delta_f = np.where(np.isfinite(gmed_f) & np.isfinite(gmed_r), gmed_f - gmed_r, np.nan)
            panels += [
                (gmed_f,  "RdYlBu_r", mag_norm,   f"{band_name} mock (≥{min_det} det)"),
                (delta_f, "RdBu_r",   delta_norm, f"{band_name} Δmag filtered − {primary_cat['label']}"),
            ]

        for col, (data, cmap, norm, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(z_bins, logM_bins, np.where(np.isfinite(data), data, np.nan).T,
                               cmap=cmap, norm=norm)
            cb_label = "ΔAB mag" if "Δ" in title else "AB mag"
            plt.colorbar(im, ax=ax, label=cb_label)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("z"); ax.set_ylabel(r"$\log M_*$")

    fig.tight_layout()
    out = outdir / "mag_grid.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_ssfr_vs_z(all_cats, mock_params, det_mask, min_det, outdir):
    """sSFR vs redshift: real catalogs vs atlas mock (all | detected subset).

    Shows median ± p16/p84 bands. Helps diagnose whether atlas has the
    right star-forming population at each redshift.
    """
    z_bins = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    z_cen  = 0.5 * (z_bins[:-1] + z_bins[1:])

    def _ssfr_profile(ssfr, z):
        med = np.full(len(z_cen), np.nan)
        p16 = np.full(len(z_cen), np.nan)
        p84 = np.full(len(z_cen), np.nan)
        for i, (zlo, zhi) in enumerate(zip(z_bins[:-1], z_bins[1:])):
            m = np.isfinite(ssfr) & np.isfinite(z) & (z >= zlo) & (z < zhi) & (ssfr > -15) & (ssfr < 0)
            if m.sum() >= 5:
                med[i], p16[i], p84[i] = np.percentile(ssfr[m], [50, 16, 84])
        return med, p16, p84

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    for cat in all_cats:
        ssfr = cat.get("sSFR", cat["logSFR"] - cat["logM"])
        style = CAT_STYLES.get(cat["_key"], dict(color="gray", ls="-", lw=1.5))
        med, p16, p84 = _ssfr_profile(ssfr, cat["z"])
        ax.fill_between(z_cen, p16, p84, alpha=0.12, color=style["color"])
        ax.plot(z_cen, med, label=cat["label"], **style)

    # mock atlas — all galaxies
    mock_ssfr = mock_params["logSFR"] - mock_params["logM"]
    med_m, p16_m, p84_m = _ssfr_profile(mock_ssfr, mock_params["z"])
    ax.fill_between(z_cen, p16_m, p84_m, alpha=0.10, color="tomato")
    ax.plot(z_cen, med_m, color="tomato", lw=1.5, ls="--", label="mock atlas (all)")

    # mock atlas — detected subset
    if det_mask is not None and min_det > 0:
        med_f, p16_f, p84_f = _ssfr_profile(mock_ssfr[det_mask], mock_params["z"][det_mask])
        ax.fill_between(z_cen, p16_f, p84_f, alpha=0.15, color="crimson")
        ax.plot(z_cen, med_f, color="crimson", lw=2.5, ls="-",
                label=f"mock atlas (≥{min_det} det bands)")

    # main-sequence reference from the SBI simulator
    z_ref = np.linspace(0, 5, 200)
    def _ms_ssfr(z):
        coeff = np.where(z < 1, 1.0, np.where(z < 2, 2.0, 2.8))
        return -10.0 + coeff * np.log10(1 + z)
    ms = _ms_ssfr(z_ref)
    ax.plot(z_ref, ms, color="black", lw=1.2, ls=":", alpha=0.6, label="Schreiber+15 MS (logM=9)")
    ax.fill_between(z_ref, ms - 0.3, ms + 0.3, color="black", alpha=0.06)

    ax.set_xlabel("Redshift z", fontsize=12)
    ax.set_ylabel(r"$\log(\mathrm{sSFR}\;[\mathrm{yr}^{-1}])$", fontsize=12)
    ax.set_title("sSFR vs redshift — real catalogs (median ± p16/p84) vs mock atlas", fontsize=11)
    ax.set_ylim(-13, -5)
    ax.set_xlim(0, 5)
    ax.axhline(-9, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outdir / "ssfr_vs_z.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-name",    default="atlas_obs_euclid_north_validate_50000_Nparam_2.dbatlas",
                   help="Atlas filename in library/ (exact, with .dbatlas)")
    p.add_argument("--phot-type",     default="templfit")
    p.add_argument("--catalogs",      nargs="+",
                   default=["cosmos_deep", "cosmos_web", "desi"],
                   choices=list(CATALOG_SPECS.keys()),
                   help="Obs catalogs to load")
    p.add_argument("--primary-cat",   default="cosmos_deep",
                   choices=list(CATALOG_SPECS.keys()),
                   help="Primary catalog for sigma_vs_mag and sigma_grid panels")
    p.add_argument("--min-det-bands", type=int, default=3,
                   help="Training filter: keep mock galaxies with SNR>=3 in at least N bands "
                        "(0 = no filter, 3 = matches typical inference selection)")
    p.add_argument("--outdir",        default="sbi-logs/diagnose_sigma")
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"diagnose_sigma_vs_mag")
    print(f"  Atlas        : {args.atlas_name}")
    print(f"  Phot         : {args.phot_type}")
    print(f"  Catalogs     : {args.catalogs}")
    print(f"  min-det-bands: {args.min_det_bands}  (0=no filter)")
    print(f"  Outdir       : {outdir}")
    print("=" * 60)

    print("\nLoading obs catalogs...")
    all_cats = []
    for name in args.catalogs:
        cat = load_catalog(name, args.phot_type)
        cat["_key"] = name
        all_cats.append(cat)

    primary_cat = next(c for c in all_cats if c["_key"] == args.primary_cat)

    theta_dict, obs_mag_true, mock_mag, mock_sigma, model, det_mask = \
        load_atlas_and_inject_noise(args)

    print("\nGenerating plots...")
    plot_sigma_vs_mag(primary_cat, theta_dict, mock_mag, mock_sigma, model, outdir)
    plot_sigma_vs_z(all_cats, theta_dict, mock_mag, mock_sigma,
                    det_mask, args.min_det_bands, outdir)
    plot_coverage(all_cats, theta_dict, outdir)
    plot_sigma_grid(primary_cat, theta_dict, mock_mag, mock_sigma,
                    det_mask, args.min_det_bands, outdir)
    plot_mag_vs_z(all_cats, theta_dict, obs_mag_true, mock_mag,
                  det_mask, args.min_det_bands, outdir)
    plot_mag_grid(primary_cat, theta_dict, obs_mag_true, mock_mag,
                  det_mask, args.min_det_bands, outdir)
    plot_ssfr_vs_z(all_cats, theta_dict, det_mask, args.min_det_bands, outdir)

    print(f"\nDone. All plots → {outdir}/")


if __name__ == "__main__":
    main()
