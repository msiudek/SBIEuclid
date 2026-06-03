"""
diagnose_sigma_vs_mag.py — Atlas noise/magnitude diagnostic.

Loads a dbatlas, injects observational noise, applies a detection filter,
and produces diagnostic plots comparing mock vs real catalogs.

Plots saved to --outdir:
    coverage.png          — 2D histograms N(z,logM*) and N(z,logsSFR) for real+mock
    mag_grid.png          — median mag in (z,logM*) cells: real/mock/delta for VIS+NISP-H
    mag_vs_z.png          — median mag vs z with percentile bands for VIS+NISP-H
    sigma_grid.png        — median sigma in (z,logM*) cells: real/mock for VIS+NISP-H
    sigma_vs_z.png        — median sigma vs z with percentile bands for VIS+NISP-H
    sigma_vs_mag_VIS.png  — scatter sigma vs mag colored by z/logM/logSFR/logsSFR
    sigma_vs_mag_NISP_H.png
    ssfr_vs_z.png         — median sSFR vs z: real catalogs + mock all + mock detected

Usage
-----
python examples/diagnose_sigma_vs_mag.py \\
    --atlas atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas \\
    --phot-type templfit \\
    --min-det-bands 3 \\
    --outdir sbi-logs/diagnose_v1.0 \\
    2>&1 | tee sbi-logs/diagnose_v1.0.log
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbipix.utils.sed_utils import load_filter_metadata, flux_ujy_to_mag

ROOT    = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"

FILTER_META  = load_filter_metadata("filters_to_use.dat", filt_dir=str(OBS_DIR))
FILTER_SHORT = [m["short"]    for m in FILTER_META]
FILTER_STEMS = [m["col_stem"] for m in FILTER_META]
N_FILT = len(FILTER_META)

VIS_IDX = FILTER_SHORT.index("VIS")
H_IDX   = FILTER_SHORT.index("NISP-H")

# 2D cell edges used in mag_grid / sigma_grid
Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(5.0, 12.5, 0.5)
Z_CEN   = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
M_CEN   = 0.5 * (M_EDGES[:-1] + M_EDGES[1:])

# z-profile bins (finer, for mag_vs_z / sigma_vs_z)
ZP_EDGES = np.arange(0.0, 5.25, 0.25)
ZP_CEN   = 0.5 * (ZP_EDGES[:-1] + ZP_EDGES[1:])

CATALOG_INFO = {
    "cosmos_deep": {
        "path":    OBS_DIR / "COSMOS_DEEP_PHZ.fits",
        "z_col":   "PHZ_PP_MEDIAN_REDSHIFT",
        "mass_col":"PHZ_PP_MEDIAN_STELLARMASS",
        "sfr_col": "PHZ_PP_MEDIAN_SFR",
        "sfr_log": False,
        "label":   "COSMOS-Deep",
        "color":   "#1f77b4",
    },
    "cosmos_web": {
        "path":    OBS_DIR / "COSMOS-Web" / "matched_euclid_cosmosweb.fits",
        "z_col":   "z_lephare",
        "mass_col":"logM_lephare",
        "sfr_col": "logSFR_lephare",
        "sfr_log": True,
        "label":   "COSMOS-Web",
        "color":   "#ff7f0e",
    },
}


# ── helpers ────────────────────────────────────────────────────────────────

def build_phot_col(stem, phot_type, err=False):
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"
    return f"{prefix}_{stem}_{phot_type}_aper"


def load_catalog(cat_key, phot_type):
    from astropy.table import Table
    info = CATALOG_INFO[cat_key]
    cat  = Table.read(info["path"])
    print(f"  [{cat_key}] {len(cat):,} rows")

    z    = np.array(cat[info["z_col"]],    dtype=float) if info["z_col"]    in cat.colnames else np.full(len(cat), np.nan)
    logM = np.array(cat[info["mass_col"]], dtype=float) if info["mass_col"] in cat.colnames else np.full(len(cat), np.nan)

    logSFR = np.full(len(cat), np.nan)
    if info["sfr_col"] in cat.colnames:
        raw = np.array(cat[info["sfr_col"]], dtype=float)
        logSFR = raw if info["sfr_log"] else np.where(raw > 0, np.log10(raw), np.nan)

    logsSFR = np.where(np.isfinite(logM) & (logM > 0) & np.isfinite(logSFR),
                       logSFR - logM, np.nan)

    flux_cols    = [build_phot_col(s, phot_type, err=False) for s in FILTER_STEMS]
    fluxerr_cols = [build_phot_col(s, phot_type, err=True)  for s in FILTER_STEMS]

    if any(c not in cat.colnames for c in flux_cols):
        missing = [c for c in flux_cols if c not in cat.colnames]
        print(f"    WARNING: missing flux cols: {missing[:3]}; skipping")
        return None

    flux    = np.column_stack([np.array(cat[c], dtype=float) for c in flux_cols])
    fluxerr = np.column_stack([np.array(cat[c], dtype=float) for c in fluxerr_cols])

    return {"flux": flux, "fluxerr": fluxerr,
            "z": z, "logM": logM, "logSFR": logSFR, "logsSFR": logsSFR,
            "label": info["label"], "color": info["color"]}


def real_mag_sigma(flux_col, fluxerr_col):
    """Per-galaxy AB magnitude and sigma_mag from flux/fluxerr (µJy)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.isfinite(flux_col) & (flux_col > 0),
                       -2.5 * np.log10(np.maximum(flux_col, 1e-30) / 3631e6), np.nan)
        sig = np.where(np.isfinite(flux_col) & (flux_col > 0) &
                       np.isfinite(fluxerr_col) & (fluxerr_col > 0),
                       (2.5 / np.log(10)) * np.abs(fluxerr_col / flux_col), np.nan)
    return mag, sig


def cell_median(values, z, logm, min_count=3):
    """Median of values in (Z_EDGES, M_EDGES) cells → shape (n_m, n_z)."""
    nz, nm = len(Z_EDGES)-1, len(M_EDGES)-1
    grid = np.full((nm, nz), np.nan)
    for iz in range(nz):
        for im in range(nm):
            mask = ((z  >= Z_EDGES[iz]) & (z  < Z_EDGES[iz+1]) &
                    (logm >= M_EDGES[im]) & (logm < M_EDGES[im+1]) &
                    np.isfinite(values))
            if mask.sum() >= min_count:
                grid[im, iz] = np.nanmedian(values[mask])
    return grid


def cell_count(z, logm):
    """Count in (Z_EDGES, M_EDGES) cells → shape (n_m, n_z)."""
    nz, nm = len(Z_EDGES)-1, len(M_EDGES)-1
    grid = np.zeros((nm, nz), dtype=int)
    for iz in range(nz):
        for im in range(nm):
            mask = ((z  >= Z_EDGES[iz]) & (z  < Z_EDGES[iz+1]) &
                    (logm >= M_EDGES[im]) & (logm < M_EDGES[im+1]))
            grid[im, iz] = mask.sum()
    return grid


def zprofile(values, z, pcts=(16, 50, 84)):
    """Percentile profiles in ZP_EDGES bins. Returns (n_bins, n_pcts) where nan = no data."""
    result = np.full((len(ZP_CEN), len(pcts)), np.nan)
    for i, (lo, hi) in enumerate(zip(ZP_EDGES[:-1], ZP_EDGES[1:])):
        m = np.isfinite(values) & np.isfinite(z) & (z >= lo) & (z < hi)
        if m.sum() >= 5:
            result[i] = np.percentile(values[m], pcts)
    return result


def _heatmap(ax, grid, x_edges, y_edges, cmap, norm, min_count_grid=None):
    """pcolormesh with NaN for empty cells."""
    disp = grid.copy().astype(float)
    if min_count_grid is not None:
        disp[min_count_grid < 1] = np.nan
    pcm = ax.pcolormesh(x_edges, y_edges, disp, cmap=cmap, norm=norm)
    return pcm


# ── coverage ───────────────────────────────────────────────────────────────

def plot_coverage(real_cats, mock_z, mock_logM, mock_logSFR, outdir):
    z_e  = np.linspace(0, 5, 51)
    m_e  = np.linspace(5, 12, 51)
    ss_e = np.linspace(-14, -7, 51)

    sources = []
    for key in ["cosmos_deep", "cosmos_web"]:
        if key in real_cats and real_cats[key] is not None:
            c = real_cats[key]
            sources.append((c["label"], c["color"], c["z"], c["logM"], c["logsSFR"]))

    mock_logsSFR = np.where(np.isfinite(mock_logM) & (mock_logM > 0) & np.isfinite(mock_logSFR),
                            mock_logSFR - mock_logM, np.nan)
    sources.append(("mock", "k", mock_z, mock_logM, mock_logsSFR))

    fig, axes = plt.subplots(2, len(sources), figsize=(5*len(sources), 9))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)
    norm_n = mcolors.LogNorm(vmin=1, vmax=1e4)

    for col, (lbl, col_c, z, logm, logssfr) in enumerate(sources):
        ok_m  = np.isfinite(z) & (z >= 0) & np.isfinite(logm)
        ok_ss = np.isfinite(z) & (z >= 0) & np.isfinite(logssfr)

        h_m,  _, _ = np.histogram2d(z[ok_m],  logm[ok_m],    bins=[z_e, m_e])
        h_ss, _, _ = np.histogram2d(z[ok_ss], logssfr[ok_ss], bins=[z_e, ss_e])
        h_m  = np.where(h_m  > 0, h_m,  np.nan)
        h_ss = np.where(h_ss > 0, h_ss, np.nan)

        pcm0 = axes[0, col].pcolormesh(z_e, m_e,  h_m.T,  cmap="viridis", norm=norm_n)
        pcm1 = axes[1, col].pcolormesh(z_e, ss_e, h_ss.T, cmap="viridis", norm=norm_n)
        for ax in [axes[0,col], axes[1,col]]:
            ax.set_xlabel("z", fontsize=10)
            ax.set_xlim(0, 5)
        axes[0, col].set_ylabel(r"$\log M_*$", fontsize=10)
        axes[1, col].set_ylabel(r"$\log$ sSFR", fontsize=10)
        axes[0, col].set_ylim(5, 12)
        axes[1, col].set_ylim(-14, -7)
        axes[0, col].set_title(lbl, fontsize=11)
        plt.colorbar(pcm0, ax=axes[0,col], label="N")
        plt.colorbar(pcm1, ax=axes[1,col], label="N")

    fig.suptitle("Parameter coverage", fontsize=13)
    fig.tight_layout()
    out = outdir / "coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── mag_grid ───────────────────────────────────────────────────────────────

def plot_mag_grid(real_cd, noiseless_mag, noisy_mag, det_mask,
                  mock_z, mock_logM, outdir):
    """2 rows (VIS, NISP-H) × 6 cols: real | noiseless | noisy | Δnoisy | det | Δdet."""
    bands = [(VIS_IDX, "VIS"), (H_IDX, "NISP-H")]
    col_titles = ["COSMOS-Deep (real)",
                  "mock noiseless SED",
                  "mock noisy",
                  "Δmag noisy − COSMOS-Deep",
                  f"mock (≥{MIN_DET_BANDS} det)",
                  f"Δmag filtered − COSMOS-Deep"]

    mag_norm  = mcolors.Normalize(vmin=18, vmax=28)
    diff_norm = mcolors.Normalize(vmin=-3, vmax=3)
    mag_cmap  = "plasma"
    diff_cmap = "RdBu_r"

    fig, axes = plt.subplots(2, 6, figsize=(26, 8))
    fig.suptitle(
        "Median AB magnitude in (z, log M*) cells — COSMOS-Deep | mock noiseless | mock noisy | Δ(mock−real)\n"
        "Δ > 0 = mock fainter → SED issue;  Δ = 0 + high σ → noise model issue",
        fontsize=10)

    for row, (fi, fname) in enumerate(bands):
        # real mag in cells
        real_z = real_cd["z"]; real_logM = real_cd["logM"]
        real_flux = real_cd["flux"][:, fi]
        ok_r = (real_flux > 0) & np.isfinite(real_flux) & np.isfinite(real_z) & np.isfinite(real_logM)
        real_mag_all = np.where(ok_r, flux_ujy_to_mag(np.maximum(real_flux, 1e-30)), np.nan)
        grid_real  = cell_median(real_mag_all, real_z, real_logM)

        # mock: noiseless magnitude in cells
        nl_mag = flux_ujy_to_mag(np.maximum(noiseless_mag[:, fi], 1e-30))
        ok_nl  = noiseless_mag[:, fi] > 0
        nl_mag_use = np.where(ok_nl, nl_mag, np.nan)
        grid_nl = cell_median(nl_mag_use, mock_z, mock_logM)

        # mock: noisy mag in cells (all, non-det = 99 → exclude)
        nm_use = np.where(noisy_mag[:, fi] < 98.0, noisy_mag[:, fi], np.nan)
        grid_noisy = cell_median(nm_use, mock_z, mock_logM)

        # mock: noisy mag (det only)
        nm_det = np.where(det_mask & (noisy_mag[:, fi] < 98.0), noisy_mag[:, fi], np.nan)
        grid_det = cell_median(nm_det, mock_z, mock_logM)

        grids = [grid_real, grid_nl, grid_noisy,
                 grid_noisy - grid_real, grid_det,
                 grid_det - grid_real]
        cmaps = [mag_cmap, mag_cmap, mag_cmap, diff_cmap, mag_cmap, diff_cmap]
        norms = [mag_norm,  mag_norm,  mag_norm,  diff_norm,  mag_norm,  diff_norm]

        for col, (g, cm, nm) in enumerate(zip(grids, cmaps, norms)):
            ax = axes[row, col]
            cnt = cell_count(
                real_z if col == 0 else mock_z,
                real_logM if col == 0 else mock_logM,
            )
            pcm = _heatmap(ax, g, Z_EDGES, M_EDGES, cm, nm, min_count_grid=cnt)
            plt.colorbar(pcm, ax=ax, label="AB mag" if col < 3 or col == 4 else "Δ AB mag")
            ax.set_xlim(0, 5); ax.set_ylim(5, 12)
            ax.set_xlabel("z", fontsize=8)
            ax.set_ylabel(r"$\log M_*$", fontsize=8)
            ax.set_title(f"{fname} {col_titles[col]}", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = outdir / "mag_grid.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── mag_vs_z ───────────────────────────────────────────────────────────────

def plot_mag_vs_z(real_cats, noiseless_mag, noisy_mag, det_mask, mock_z, outdir):
    """Median magnitude vs redshift for VIS and NISP-H."""
    bands = [(VIS_IDX, "VIS"), (H_IDX, "NISP-H")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Median AB magnitude vs redshift\n"
        "mock above real → SED too faint (SED issue);  mock ≈ real but σ high → noise model issue",
        fontsize=11)

    for ax, (fi, fname) in zip(axes, bands):
        # real catalogs
        for key in ["cosmos_deep", "cosmos_web"]:
            c = real_cats.get(key)
            if c is None:
                continue
            f_ok = c["flux"][:, fi] > 0
            mag_r = np.where(f_ok, flux_ujy_to_mag(np.maximum(c["flux"][:, fi], 1e-30)), np.nan)
            ok = np.isfinite(mag_r) & np.isfinite(c["z"]) & (c["z"] > 0)
            prof = zprofile(mag_r[ok], c["z"][ok])
            ls = "-" if key == "cosmos_deep" else "--"
            ax.plot(ZP_CEN, prof[:, 1], color=c["color"], ls=ls, lw=2, label=c["label"])
            ax.fill_between(ZP_CEN, prof[:, 0], prof[:, 2], alpha=0.2, color=c["color"])

        ok_z = np.isfinite(mock_z) & (mock_z > 0)

        # mock noiseless
        nl_mag = np.where(noiseless_mag[:, fi] > 0,
                          flux_ujy_to_mag(np.maximum(noiseless_mag[:, fi], 1e-30)), np.nan)
        ok_nl = ok_z & np.isfinite(nl_mag)
        prof_nl = zprofile(nl_mag[ok_nl], mock_z[ok_nl])
        ax.plot(ZP_CEN, prof_nl[:, 1], color="purple", ls=":", lw=1.5, label="mock (noiseless SED)")
        ax.fill_between(ZP_CEN, prof_nl[:, 0], prof_nl[:, 2], alpha=0.10, color="purple")

        # mock noisy all
        nm_all = np.where(noisy_mag[:, fi] < 98.0, noisy_mag[:, fi], np.nan)
        ok_na  = ok_z & np.isfinite(nm_all)
        prof_na = zprofile(nm_all[ok_na], mock_z[ok_na])
        ax.plot(ZP_CEN, prof_na[:, 1], color="#e6693a", ls="--", lw=1.5, label="mock noisy (all)")
        ax.fill_between(ZP_CEN, prof_na[:, 0], prof_na[:, 2], alpha=0.10, color="#e6693a")

        # mock noisy detected
        nm_det = np.where(det_mask & (noisy_mag[:, fi] < 98.0), noisy_mag[:, fi], np.nan)
        ok_nd  = ok_z & np.isfinite(nm_det)
        prof_nd = zprofile(nm_det[ok_nd], mock_z[ok_nd])
        ax.plot(ZP_CEN, prof_nd[:, 1], color="#c0392b", ls="-", lw=2.5,
                label=f"mock (≥{MIN_DET_BANDS} det bands)")
        ax.fill_between(ZP_CEN, prof_nd[:, 0], prof_nd[:, 2], alpha=0.15, color="#c0392b")

        ax.set_xlabel("Redshift z", fontsize=11)
        ax.set_ylabel("AB magnitude (median, lower = brighter)", fontsize=10)
        ax.set_title(fname, fontsize=12)
        ax.set_xlim(0, 5)
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = outdir / "mag_vs_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── sigma_grid ─────────────────────────────────────────────────────────────

def plot_sigma_grid(real_cd, mock_sigma, det_mask, mock_z, mock_logM, outdir):
    """2 rows (VIS, NISP-H) × 6 cols: real σ | real N | mock-all σ | N | mock-det σ | N."""
    bands = [(VIS_IDX, "VIS"), (H_IDX, "NISP-H")]

    sig_norm = mcolors.LogNorm(vmin=0.01, vmax=10)
    n_norm   = mcolors.LogNorm(vmin=1,    vmax=1e4)
    sig_cmap = "RdYlGn_r"
    n_cmap   = "viridis"

    fig, axes = plt.subplots(2, 6, figsize=(26, 8))
    fig.suptitle(
        r"Median $\sigma_{\rm mag}$ in (z, log M*) cells — COSMOS-Deep | mock-all | mock-filtered"
        f" |  filtered: ≥{MIN_DET_BANDS} det bands",
        fontsize=10)

    for row, (fi, fname) in enumerate(bands):
        real_z   = real_cd["z"];  real_logM = real_cd["logM"]
        r_f      = real_cd["flux"][:, fi];  r_fe = real_cd["fluxerr"][:, fi]
        _, real_sig = real_mag_sigma(r_f, r_fe)
        ok_r = np.isfinite(real_z) & np.isfinite(real_logM)

        grid_real_sig = cell_median(real_sig, real_z, real_logM)
        cnt_real      = cell_count(real_z[ok_r], real_logM[ok_r])

        # mock all
        mock_s_all  = mock_sigma[:, fi]
        ok_m        = np.isfinite(mock_z) & np.isfinite(mock_logM)
        grid_mall_s = cell_median(mock_s_all, mock_z, mock_logM)
        cnt_mall    = cell_count(mock_z[ok_m], mock_logM[ok_m])

        # mock detected
        mock_s_det  = np.where(det_mask, mock_sigma[:, fi], np.nan)
        ok_det      = ok_m & det_mask
        grid_mdet_s = cell_median(mock_s_det, mock_z, mock_logM)
        cnt_mdet    = cell_count(mock_z[ok_det], mock_logM[ok_det])

        data_cols = [
            (grid_real_sig, sig_cmap, sig_norm, "median σ"),
            (cnt_real.astype(float), n_cmap, n_norm, "N"),
            (grid_mall_s, sig_cmap, sig_norm, "median σ"),
            (cnt_mall.astype(float), n_cmap, n_norm, "N"),
            (grid_mdet_s, sig_cmap, sig_norm, "median σ"),
            (cnt_mdet.astype(float), n_cmap, n_norm, "N"),
        ]
        subtitles = [f"{fname} COSMOS-Deep: median σ",
                     f"{fname} COSMOS-Deep: N",
                     f"{fname} mock (all): median σ",
                     f"{fname} mock (all): N",
                     f"{fname} mock (≥{MIN_DET_BANDS} det): median σ",
                     f"{fname} mock (≥{MIN_DET_BANDS} det): N"]

        for col, ((g, cm, nm, lbl), stitle) in enumerate(zip(data_cols, subtitles)):
            ax = axes[row, col]
            g_disp = g.copy().astype(float)
            g_disp[g_disp <= 0] = np.nan
            pcm = ax.pcolormesh(Z_EDGES, M_EDGES, g_disp, cmap=cm, norm=nm)
            plt.colorbar(pcm, ax=ax, label=lbl)
            ax.set_xlim(0, 5); ax.set_ylim(5, 12)
            ax.set_xlabel("z", fontsize=8)
            ax.set_ylabel(r"$\log M_*$", fontsize=8)
            ax.set_title(stitle, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = outdir / "sigma_grid.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── sigma_vs_z ─────────────────────────────────────────────────────────────

def plot_sigma_vs_z(real_cats, mock_sigma, det_mask, mock_z, outdir):
    """Median sigma_mag vs redshift for VIS and NISP-H (log y-scale)."""
    bands = [(VIS_IDX, "VIS"), (H_IDX, "NISP-H")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(r"Median $\sigma_{\rm mag}$ vs redshift", fontsize=12)

    for ax, (fi, fname) in zip(axes, bands):
        for key in ["cosmos_deep", "cosmos_web"]:
            c = real_cats.get(key)
            if c is None:
                continue
            _, sig_r = real_mag_sigma(c["flux"][:, fi], c["fluxerr"][:, fi])
            ok = np.isfinite(sig_r) & (sig_r > 0) & np.isfinite(c["z"]) & (c["z"] > 0)
            prof = zprofile(sig_r[ok], c["z"][ok])
            ls = "-" if key == "cosmos_deep" else "--"
            ax.plot(ZP_CEN, prof[:, 1], color=c["color"], ls=ls, lw=2, label=c["label"])
            ax.fill_between(ZP_CEN, np.maximum(prof[:, 0], 1e-4),
                            np.maximum(prof[:, 2], 1e-4), alpha=0.2, color=c["color"])

        ok_z = np.isfinite(mock_z) & (mock_z > 0)

        # mock all
        s_all = mock_sigma[:, fi]
        ok_all = ok_z & np.isfinite(s_all) & (s_all > 0)
        prof_a = zprofile(s_all[ok_all], mock_z[ok_all])
        ax.plot(ZP_CEN, prof_a[:, 1], color="#e6693a", ls="--", lw=2, label="mock (all)")
        ax.fill_between(ZP_CEN, np.maximum(prof_a[:, 0], 1e-4),
                        np.maximum(prof_a[:, 2], 1e-4), alpha=0.15, color="#e6693a")

        # mock detected
        s_det = np.where(det_mask, mock_sigma[:, fi], np.nan)
        ok_det = ok_z & np.isfinite(s_det) & (s_det > 0)
        prof_d = zprofile(s_det[ok_det], mock_z[ok_det])
        ax.plot(ZP_CEN, prof_d[:, 1], color="#c0392b", ls="-", lw=2.5,
                label=f"mock (≥{MIN_DET_BANDS} det bands)")
        ax.fill_between(ZP_CEN, np.maximum(prof_d[:, 0], 1e-4),
                        np.maximum(prof_d[:, 2], 1e-4), alpha=0.15, color="#c0392b")

        ax.set_yscale("log")
        ax.set_xlabel("Redshift z", fontsize=11)
        ax.set_ylabel(r"$\sigma_{\rm mag}$", fontsize=11)
        ax.set_title(fname, fontsize=12)
        ax.set_xlim(0, 5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = outdir / "sigma_vs_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── sigma_vs_mag ───────────────────────────────────────────────────────────

def plot_sigma_vs_mag(fi, fname, real_cd, noiseless_mag, mock_sigma, mock_z,
                      mock_logM, mock_logSFR, det_mask, outdir):
    """4×2 scatter of sigma_mag vs mag, colored by z / logM* / logSFR / logsSFR."""
    mock_logsSFR = np.where(np.isfinite(mock_logM) & (mock_logM > 0) & np.isfinite(mock_logSFR),
                            mock_logSFR - mock_logM, np.nan)
    real_logsSFR = real_cd["logsSFR"]

    rows = [
        ("Redshift z",    real_cd["z"],      mock_z,       "viridis",  (0, 4.5),   None),
        (r"$\log M_*$",   real_cd["logM"],   mock_logM,    "plasma",   (7, 11.5),  None),
        ("log SFR",       real_cd["logSFR"], mock_logSFR,  "YlOrRd",   (-3, 2.5),  None),
        ("log sSFR",      real_logsSFR,      mock_logsSFR, "RdPu_r",   (-13.5, -8.5), None),
    ]

    # real data
    real_f  = real_cd["flux"][:, fi]
    real_fe = real_cd["fluxerr"][:, fi]
    real_mag_vals, real_sig_vals = real_mag_sigma(real_f, real_fe)
    ok_real = np.isfinite(real_mag_vals) & np.isfinite(real_sig_vals) & (real_sig_vals > 0)

    # mock: use all galaxies, noiseless mag + noise model sigma
    nl_mag   = np.where(noiseless_mag[:, fi] > 0,
                        flux_ujy_to_mag(np.maximum(noiseless_mag[:, fi], 1e-30)), np.nan)
    mock_sig = mock_sigma[:, fi]
    ok_mock  = np.isfinite(nl_mag) & np.isfinite(mock_sig) & (mock_sig > 0)

    # downsample real for speed (max 80k points)
    rng = np.random.default_rng(42)
    real_idx = np.where(ok_real)[0]
    if len(real_idx) > 80000:
        real_idx = rng.choice(real_idx, 80000, replace=False)
    mock_idx = np.where(ok_mock)[0]

    n_real = len(real_idx)
    n_mock = len(mock_idx)

    fig, axes = plt.subplots(4, 2, figsize=(12, 22))
    fig.suptitle(
        f"{fname} — $\\sigma_{{\\rm mag}}$ vs magnitude\n"
        "Left: COSMOS-Deep    Right: mock atlas",
        fontsize=12)

    dashed_mags = [22, 24, 26, 28]

    for row, (clabel, real_c, mock_c, cmap, vlim, _) in enumerate(rows):
        for col, (idx, mag_v, sig_v, c_vals, N_label) in enumerate([
            (real_idx, real_mag_vals, real_sig_vals, real_c, f"COSMOS-Deep · {clabel} (N={n_real:,})"),
            (mock_idx, nl_mag,        mock_sig,       mock_c, f"mock · {clabel} (N={n_mock:,})"),
        ]):
            ax = axes[row, col]
            c_use = c_vals[idx] if c_vals is not None else None
            ok_c  = np.isfinite(c_use) if c_use is not None else np.ones(len(idx), dtype=bool)
            idx2  = idx[ok_c]
            cv    = c_use[ok_c]

            sc = ax.scatter(mag_v[idx2], sig_v[idx2],
                            c=cv, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                            s=0.5, alpha=0.4, rasterized=True)
            plt.colorbar(sc, ax=ax, label=clabel)

            for dm in dashed_mags:
                ax.axvline(dm, color="gray", lw=0.5, ls="--", alpha=0.5)

            ax.set_yscale("log")
            ax.set_xlim(17, 30)
            ax.set_ylim(0.008, 12)
            ax.set_xlabel("AB magnitude", fontsize=9)
            ax.set_ylabel(r"$\sigma_{\rm mag}$", fontsize=9)
            ax.set_title(N_label, fontsize=8)
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    fname_safe = fname.replace("-", "_")
    out = outdir / f"sigma_vs_mag_{fname_safe}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── ssfr_vs_z ──────────────────────────────────────────────────────────────

def plot_ssfr_vs_z(real_cats, mock_z, mock_logM, mock_logSFR, det_mask, outdir):
    """Median sSFR vs redshift: real catalogs vs mock all + detected + Schreiber+15."""
    mock_ssfr = np.where(np.isfinite(mock_logM) & (mock_logM > 0) & np.isfinite(mock_logSFR),
                         mock_logSFR - mock_logM, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))

    for key in ["cosmos_deep", "cosmos_web"]:
        c = real_cats.get(key)
        if c is None:
            continue
        ok = (np.isfinite(c["logsSFR"]) & np.isfinite(c["z"]) & (c["z"] > 0) &
              (c["logsSFR"] > -15) & (c["logsSFR"] < 0))
        prof = zprofile(c["logsSFR"][ok], c["z"][ok])
        ls = "-" if key == "cosmos_deep" else "--"
        ax.plot(ZP_CEN, prof[:, 1], color=c["color"], ls=ls, lw=2, label=c["label"])
        ax.fill_between(ZP_CEN, prof[:, 0], prof[:, 2], alpha=0.2, color=c["color"])

    ok_m = np.isfinite(mock_z) & (mock_z > 0) & np.isfinite(mock_ssfr) & (mock_ssfr > -15)
    prof_m = zprofile(mock_ssfr[ok_m], mock_z[ok_m])
    ax.plot(ZP_CEN, prof_m[:, 1], "k--", lw=2, label="mock (all)")
    ax.fill_between(ZP_CEN, prof_m[:, 0], prof_m[:, 2], alpha=0.1, color="k")

    ok_d = ok_m & det_mask
    prof_d = zprofile(mock_ssfr[ok_d], mock_z[ok_d])
    ax.plot(ZP_CEN, prof_d[:, 1], "rs-", ms=4, lw=2, label=f"mock (≥{MIN_DET_BANDS} det)")
    ax.fill_between(ZP_CEN, prof_d[:, 0], prof_d[:, 2], alpha=0.1, color="r")

    z_ref = np.linspace(0.05, 5.0, 100)
    ms_ssfr = np.where(z_ref < 1, 1.0, np.where(z_ref < 2, 2.0, 2.8)) * np.log10(1 + z_ref) - 10.0
    ax.plot(z_ref, ms_ssfr, "r--", lw=1.5, label="Schreiber+15 MS")

    ax.set_xlabel("Redshift", fontsize=11)
    ax.set_ylabel(r"$\log_{10}(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$", fontsize=11)
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

MIN_DET_BANDS = 3   # module-level default, overridden in main()


def parse_args():
    p = argparse.ArgumentParser(description="Atlas noise/magnitude diagnostic")
    p.add_argument("--atlas", type=str,
                   default="atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas")
    p.add_argument("--phot-type", type=str, default="templfit",
                   choices=["templfit", "2fwhm", "3fwhm"])
    p.add_argument("--min-det-bands", type=int, default=3,
                   help="Min bands with SNR≥3 to count as detected (default: 3)")
    p.add_argument("--outdir", type=str, default="sbi-logs/diagnose_v1.0")
    return p.parse_args()


def main():
    global MIN_DET_BANDS

    args = parse_args()
    MIN_DET_BANDS = args.min_det_bands

    print("=" * 60)
    print("diagnose_sigma_vs_mag")
    print(f"  Atlas        : {args.atlas}")
    print(f"  Phot         : {args.phot_type}")
    print(f"  min-det-bands: {args.min_det_bands}")
    print(f"  Outdir       : {args.outdir}")
    print("=" * 60)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    noise_prefix = f"north_{args.phot_type}"

    # ── sbipix noise model ────────────────────────────────────────────
    from sbipix import sbipix
    import hickle

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
    sx.configure_noise_model(sigma_sampler="mag_lognormal",
                             detection_model="hard", observation_space="mag")
    sx.load_obs_features()
    print(f"Noise model loaded (sigma_samples: {sx.sigma_samples_obs is not None})")

    # ── load atlas ────────────────────────────────────────────────────
    atlas_path = LIB_DIR / args.atlas
    print(f"\nAtlas: {args.atlas}")
    data = hickle.load(str(atlas_path))
    key_pfx = "data/" if "data/mstar" in data else ""
    mstar = np.array(data[key_pfx + "mstar"], dtype=float)
    sfr   = np.array(data[key_pfx + "sfr"],   dtype=float)
    zval  = np.array(data[key_pfx + "zval"],  dtype=float)
    sed   = np.array(data[key_pfx + "sed"],   dtype=float)   # (n, 10) µJy

    phys = mstar > 5
    mstar = mstar[phys]; sfr = sfr[phys]; zval = zval[phys]; sed = sed[phys]
    print(f"  {phys.sum()} / {len(phys)} galaxies pass logM>5 filter")

    # noiseless magnitudes
    noiseless_mag = np.where(sed > 0, flux_ujy_to_mag(np.maximum(sed, 1e-30)), np.nan)

    # inject noise
    print("  Injecting noise...")
    sx.obs           = flux_ujy_to_mag(np.maximum(sed, 1e-30))
    sx.obs           = np.where(sed > 0, sx.obs, 99.0)
    sx.n_simulation  = len(mstar)
    sx.add_noise_nan_limit_all()

    noisy_mag  = sx.mag[:, :, 0]   # (n, n_filt), 99 = non-detection
    sigma_mag  = sx.mag[:, :, 1]   # (n, n_filt)

    detected   = (noisy_mag < 98.0) & np.isfinite(noisy_mag)
    n_det_band = np.sum(detected, axis=1)
    det_mask   = n_det_band >= args.min_det_bands

    n_pass = det_mask.sum()
    print(f"  Detected ≥{args.min_det_bands} bands: {n_pass}/{len(mstar)} "
          f"({100*n_pass/len(mstar):.1f}%)")
    for lo, hi in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
        m = (zval >= lo) & (zval < hi)
        nd = (det_mask & m).sum()
        print(f"    z=[{lo},{hi}): {nd}/{m.sum()} ({100*nd/max(m.sum(),1):.1f}%)")

    # ── load real catalogs ────────────────────────────────────────────
    print("\nLoading real catalogs...")
    real_cats = {}
    for key in ["cosmos_deep", "cosmos_web"]:
        c = load_catalog(key, args.phot_type)
        if c is not None:
            real_cats[key] = c

    real_cd = real_cats.get("cosmos_deep")
    if real_cd is None:
        raise RuntimeError("COSMOS-Deep catalog is required but could not be loaded.")

    # ── plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_coverage(real_cats, zval, mstar, sfr, outdir)
    plot_mag_grid(real_cd, sed, noisy_mag, det_mask, zval, mstar, outdir)
    plot_mag_vs_z(real_cats, sed, noisy_mag, det_mask, zval, outdir)
    plot_sigma_grid(real_cd, sigma_mag, det_mask, zval, mstar, outdir)
    plot_sigma_vs_z(real_cats, sigma_mag, det_mask, zval, outdir)
    plot_sigma_vs_mag(VIS_IDX, "VIS", real_cd, sed, sigma_mag,
                      zval, mstar, sfr, det_mask, outdir)
    plot_sigma_vs_mag(H_IDX, "NISP-H", real_cd, sed, sigma_mag,
                      zval, mstar, sfr, det_mask, outdir)
    plot_ssfr_vs_z(real_cats, zval, mstar, sfr, det_mask, outdir)

    print(f"\nDone. All plots → {args.outdir}/")


if __name__ == "__main__":
    main()
