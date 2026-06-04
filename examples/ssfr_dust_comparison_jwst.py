"""
sSFR and dust comparison: atlas prior vs COSMOS-Web observations.

Compares, per (logM, z) cell:
  1. log(sSFR) distribution — is the atlas too star-forming (too blue)?
  2. Av (dust) distribution — is the atlas under-dusty (too blue)?
  3. Stellar age distribution — is the atlas too young?

Explains why the atlas is systematically bluer than real galaxies.
Atlas dust: Av = data["dust"]   (Calzetti Av, sampled from flat prior)
Real dust:  Av = 4.05 * ebv_minchi2  (Calzetti: Av = Rv * E(B-V), Rv=4.05)

Usage:
    python examples/ssfr_dust_comparison_jwst.py \
        --atlas-name atlas_jwst_50000_Nparam_2.dbatlas \
        --outdir sbi-logs/ssfr_dust_comparison
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

ROOT    = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
CATALOG = ROOT / "obs" / "obs_properties" / "COSMOS" / "COSMOSWeb_mastercatalog_v1.fits"

FILTER_STEMS = ["f115w", "f150w", "f277w", "f444w"]

Z_EDGES = np.arange(0.0, 5.5, 0.5)
M_EDGES = np.arange(6.0, 13.0, 0.5)
CALZETTI_RV = 4.05   # Av = Rv * E(B-V) for Calzetti law


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-name", default="atlas_jwst_50000_Nparam_2.dbatlas")
    p.add_argument("--outdir",     default="sbi-logs/ssfr_dust_comparison")
    p.add_argument("--snr-min",    type=float, default=3.0)
    return p.parse_args()


def load_atlas(atlas_name):
    data   = hickle.load(str(LIB_DIR / atlas_name))
    logM   = np.array(data["mstar"], dtype=float)
    z      = np.array(data["zval"],  dtype=float)
    logSFR = np.array(data["sfr"],   dtype=float)
    dust   = np.array(data["dust"],  dtype=float)   # Av
    met    = np.array(data["met"],   dtype=float)
    return logM, z, logSFR, dust, met


def load_cosmos_web(snr_min=3.0):
    print("Loading COSMOS-Web catalog...")
    phot = Table.read(str(CATALOG), hdu=1)
    ref  = Table.read(str(CATALOG), hdu=2)

    # SNR cut on all 4 bands
    fluxes   = np.zeros((len(phot), 4), dtype=float)
    fluxerrs = np.zeros((len(phot), 4), dtype=float)
    for j, stem in enumerate(FILTER_STEMS):
        f = np.array(phot[f"flux_aper_{stem}"], dtype=float)
        e = np.array(phot[f"flux_err_aper_{stem}"], dtype=float)
        fluxes[:, j]   = f[:, 0] if f.ndim == 2 else f
        fluxerrs[:, j] = e[:, 0] if e.ndim == 2 else e

    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(fluxes / np.where(fluxerrs > 0, fluxerrs, np.nan))
    good = np.all(snr >= snr_min, axis=1)

    z_ref    = np.array(ref["zpdf_med"],    dtype=float)
    logM_ref = np.array(ref["mass_med"],    dtype=float)
    logSFR   = np.array(ref["sfr_med"],     dtype=float)
    logssfr  = np.array(ref["ssfr_med"],    dtype=float)
    ebv      = np.array(ref["ebv_minchi2"], dtype=float)
    age_med  = np.array(ref["age_med"],     dtype=float)   # log age in yr

    # ssfr_med uses -99.9 as a sentinel for undetected/quiescent galaxies
    good &= (np.isfinite(z_ref) & (z_ref > 0) &
             np.isfinite(logM_ref) & (logM_ref > 4) &
             np.isfinite(logSFR) & np.isfinite(logssfr) &
             (logssfr > -20))   # remove -99.9 sentinels

    av_real = CALZETTI_RV * np.clip(ebv[good], 0, None)   # Av from E(B-V)

    print(f"  {good.sum()} galaxies pass selection (of {len(phot)})")
    return (logM_ref[good], z_ref[good], logSFR[good],
            logssfr[good], av_real, age_med[good])


def cell_stats(values, z_arr, logM_arr, min_count=5):
    """Return median and IQR grids per (logM, z) cell."""
    nz = len(Z_EDGES) - 1
    nm = len(M_EDGES) - 1
    med  = np.full((nm, nz), np.nan)
    iqr  = np.full((nm, nz), np.nan)
    cnt  = np.zeros((nm, nz), dtype=int)
    for iz in range(nz):
        for im in range(nm):
            mask = (
                (z_arr    >= Z_EDGES[iz]) & (z_arr    < Z_EDGES[iz+1]) &
                (logM_arr >= M_EDGES[im]) & (logM_arr < M_EDGES[im+1]) &
                np.isfinite(values)
            )
            if mask.sum() >= min_count:
                med[im, iz]  = np.nanmedian(values[mask])
                iqr[im, iz]  = np.nanpercentile(values[mask], 75) - np.nanpercentile(values[mask], 25)
                cnt[im, iz]  = mask.sum()
    return med, iqr, cnt


def plot_triple_grid(grid_real, grid_atlas, label, outpath, cmap_main="RdYlBu_r",
                     delta_cmap="RdBu_r", delta_label="Δ (atlas−real)", vdelta=None):
    grid_delta = grid_atlas - grid_real
    ext = [Z_EDGES[0], Z_EDGES[-1], M_EDGES[0], M_EDGES[-1]]

    finite_vals = np.concatenate([
        grid_real[np.isfinite(grid_real)],
        grid_atlas[np.isfinite(grid_atlas)]
    ])
    vmin_c = np.nanpercentile(finite_vals, 5)
    vmax_c = np.nanpercentile(finite_vals, 95)
    if vdelta is None:
        vdelta = np.nanpercentile(np.abs(grid_delta[np.isfinite(grid_delta)]), 90)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{label}\n(+) Δ = atlas higher than real  |  (−) Δ = atlas lower than real", fontsize=11)

    im0 = axes[0].imshow(grid_real,  origin="lower", cmap=cmap_main, aspect="auto",
                          extent=ext, vmin=vmin_c, vmax=vmax_c)
    axes[0].set_title("COSMOS-Web (real)"); axes[0].set_xlabel("z"); axes[0].set_ylabel("logM*")
    plt.colorbar(im0, ax=axes[0], label=label)

    im1 = axes[1].imshow(grid_atlas, origin="lower", cmap=cmap_main, aspect="auto",
                          extent=ext, vmin=vmin_c, vmax=vmax_c)
    axes[1].set_title("Atlas (FSPS)"); axes[1].set_xlabel("z")
    plt.colorbar(im1, ax=axes[1], label=label)

    im2 = axes[2].imshow(grid_delta, origin="lower", cmap=delta_cmap, aspect="auto",
                          extent=ext, vmin=-vdelta, vmax=vdelta)
    axes[2].set_title(f"{delta_label}"); axes[2].set_xlabel("z")
    plt.colorbar(im2, ax=axes[2], label=delta_label)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ {Path(outpath).name}  median Δ={np.nanmedian(grid_delta[np.isfinite(grid_delta)]):+.3f}")


def plot_distributions(real_vals, atlas_vals, real_z, atlas_z, xlabel, title, outpath,
                       bins=50, xrange=None):
    zbins   = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    zcolors = ["steelblue", "seagreen", "darkorange", "crimson"]
    labels  = ["z<1", "1<z<2", "2<z<3", "3<z<5"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=11)

    if xrange is None:
        all_v = np.concatenate([real_vals[np.isfinite(real_vals)],
                                atlas_vals[np.isfinite(atlas_vals)]])
        xrange = (np.nanpercentile(all_v, 1), np.nanpercentile(all_v, 99))
    bins_arr = np.linspace(xrange[0], xrange[1], bins)

    for (zlo, zhi), zc, lab in zip(zbins, zcolors, labels):
        mr = (real_z  >= zlo) & (real_z  < zhi) & np.isfinite(real_vals)
        ma = (atlas_z >= zlo) & (atlas_z < zhi) & np.isfinite(atlas_vals)
        if mr.sum() < 10 or ma.sum() < 10:
            continue
        ax.hist(real_vals[mr],  bins=bins_arr, density=True, histtype="step",
                color=zc, lw=2,   label=f"{lab} real")
        ax.hist(atlas_vals[ma], bins=bins_arr, density=True, histtype="step",
                color=zc, lw=1.5, linestyle="--", label=f"{lab} atlas")

    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ {Path(outpath).name}")


def main():
    args   = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    atlas_logM, atlas_z, atlas_logSFR, atlas_dust, atlas_met = load_atlas(args.atlas_name)
    real_logM, real_z, real_logSFR, real_logssfr, real_av, real_age = load_cosmos_web(args.snr_min)

    # Derived quantities
    atlas_logssfr = atlas_logSFR - atlas_logM

    # ──────────────────────────────────────────────────────────────────────
    # 1. log(sSFR) comparison
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== log(sSFR) comparison ===")
    grid_ssfr_real,  _, cnt_r = cell_stats(real_logssfr,   real_z,   real_logM)
    grid_ssfr_atlas, _, cnt_a = cell_stats(atlas_logssfr,  atlas_z,  atlas_logM)
    grid_ssfr_real[cnt_r  < 5] = np.nan
    grid_ssfr_atlas[cnt_a < 5] = np.nan

    plot_triple_grid(grid_ssfr_real, grid_ssfr_atlas,
                     label="median log(sSFR) [yr⁻¹]",
                     outpath=outdir / "ssfr_grid.png",
                     cmap_main="RdYlBu", delta_cmap="RdBu_r",
                     delta_label="Δ log(sSFR) (atlas−real)", vdelta=1.0)

    plot_distributions(real_logssfr, atlas_logssfr, real_z, atlas_z,
                       xlabel="log(sSFR) [yr⁻¹]",
                       title="sSFR distribution: COSMOS-Web (solid) vs atlas (dashed)",
                       outpath=outdir / "ssfr_distributions.png",
                       xrange=(-14, -7))

    # ──────────────────────────────────────────────────────────────────────
    # 2. Dust Av comparison
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Dust Av comparison ===")
    grid_av_real,  _, cnt_r = cell_stats(real_av,    real_z,   real_logM)
    grid_av_atlas, _, cnt_a = cell_stats(atlas_dust, atlas_z,  atlas_logM)
    grid_av_real[cnt_r  < 5] = np.nan
    grid_av_atlas[cnt_a < 5] = np.nan

    plot_triple_grid(grid_av_real, grid_av_atlas,
                     label="median Av [mag]",
                     outpath=outdir / "dust_av_grid.png",
                     cmap_main="YlOrRd", delta_cmap="RdBu_r",
                     delta_label="Δ Av (atlas−real)", vdelta=1.5)

    plot_distributions(real_av, atlas_dust, real_z, atlas_z,
                       xlabel="Av [mag]",
                       title="Dust Av: COSMOS-Web (solid) vs atlas (dashed)\nReal: 4.05×E(B-V)  |  Atlas: Calzetti Av prior",
                       outpath=outdir / "dust_distributions.png",
                       xrange=(0, 4))

    # ──────────────────────────────────────────────────────────────────────
    # 3. Summary: median offsets vs redshift
    # ──────────────────────────────────────────────────────────────────────
    print("\n=== Summary offsets vs z ===")
    z_centers = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
    nz = len(Z_EDGES) - 1

    # Pre-compute deltas for both quantities
    def z_deltas(rv, av, rz, az):
        delta, n_r = [], []
        for iz in range(nz):
            mr = (rz >= Z_EDGES[iz]) & (rz < Z_EDGES[iz+1]) & np.isfinite(rv)
            ma = (az >= Z_EDGES[iz]) & (az < Z_EDGES[iz+1]) & np.isfinite(av)
            if mr.sum() >= 5 and ma.sum() >= 5:
                delta.append(float(np.nanmedian(av[ma]) - np.nanmedian(rv[mr])))
                n_r.append(int(mr.sum()))
            else:
                delta.append(np.nan)
                n_r.append(0)
        return np.array(delta), n_r

    d_ssfr, n_ssfr = z_deltas(real_logssfr, atlas_logssfr, real_z, atlas_z)
    d_av,   n_av   = z_deltas(real_av,      atlas_dust,    real_z, atlas_z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Atlas vs COSMOS-Web: median offsets per redshift bin\n"
                 "Positive Δ = atlas has MORE of this quantity", fontsize=11)

    for ax, delta, ns, ylabel, ylim, color in [
        (axes[0], d_ssfr, n_ssfr, "Δ log(sSFR) (atlas − real) [dex]", (-1.5, 0.5), "steelblue"),
        (axes[1], d_av,   n_av,   "Δ Av (atlas − real) [mag]",         (-0.5, 2.0), "darkorange"),
    ]:
        valid = np.isfinite(delta)
        ax.plot(z_centers[valid], delta[valid], '-o', lw=2, ms=7, color=color)
        ax.axhline(0, color='k', lw=1, ls='--')
        ax.set_xlabel("Redshift z", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, 5); ax.set_ylim(ylim)
        for zc, d, n in zip(z_centers[valid], delta[valid], np.array(ns)[valid]):
            ax.annotate(f"N={n}", (zc, d), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=7, color='gray')

    plt.tight_layout()
    out = outdir / "offset_summary_vs_z.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ {out.name}")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Print summary table
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE  (atlas − real median)")
    print("=" * 60)
    print(f"{'Quantity':<20} {'All z':>8} {'z<1':>8} {'1<z<2':>8} {'2<z<3':>8} {'z>3':>8}")
    print("-" * 60)

    for name, real_v, atlas_v in [
        ("log(sSFR)",  real_logssfr, atlas_logssfr),
        ("Av [mag]",   real_av,      atlas_dust),
    ]:
        row = f"{name:<20}"
        for zlo, zhi in [(0, 99), (0, 1), (1, 2), (2, 3), (3, 9)]:
            mr = (real_z  >= zlo) & (real_z  < zhi) & np.isfinite(real_v)
            ma = (atlas_z >= zlo) & (atlas_z < zhi) & np.isfinite(atlas_v)
            if mr.sum() >= 5 and ma.sum() >= 5:
                d = np.nanmedian(atlas_v[ma]) - np.nanmedian(real_v[mr])
                row += f" {d:>+8.3f}"
            else:
                row += f" {'---':>8}"
        print(row)

    print("=" * 60)
    print()
    print("Interpretation:")
    print("  Δ log(sSFR) > 0 → atlas too star-forming → too many young hot stars → atlas too BLUE")
    print("  Δ Av < 0        → atlas under-dusty → less reddening → atlas too BLUE")
    print("  Both effects together → systematic color mismatch → explains mass bias")


if __name__ == "__main__":
    main()
