"""
DECISIVE diagnostic: M/L (and SFR/L) vs observed color — ATLAS vs REAL overlaid.

The one quantity SBI actually learns: at a fixed observed color and z, what M/L
does the training atlas assign? If the FSPS atlas sits at higher M/L than real
galaxies at the same color, every inferred mass is biased high. That offset (dex)
reads directly as the mass bias — no retraining, no inference.

Works for both surveys (--survey euclid|jwst). Runs against the v3 atlases.

Euclid:  atlas 10-band SED; anchor=NISP-H; color=VIS−H; catalog matched_euclid_cosmosweb.fits
JWST:    atlas 4-band  SED; anchor=F444W;  color=F277W−F444W; catalog COSMOSWeb_mastercatalog

Usage:
    python examples/ml_color_atlas_vs_real.py --survey euclid \
        --atlas-name atlas_euclid_v3_100k_100000_Nparam_2.dbatlas \
        --outdir sbi-logs/ml_color_euclid_v3

    python examples/ml_color_atlas_vs_real.py --survey jwst \
        --atlas-name atlas_jwst_v3_50000_Nparam_2.dbatlas \
        --outdir sbi-logs/ml_color_jwst_v3
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

ROOT    = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
COSMO   = FlatLambdaCDM(H0=70, Om0=0.3)
Z_BINS  = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]

# Per-survey configuration
SURVEY = {
    "euclid": {
        "names":      ["NISP-H", "NISP-J", "NISP-Y", "VIS", "HSC-g", "HSC-z",
                       "DECam-g", "DECam-r", "DECam-i", "DECam-z"],
        "red_idx":    0,   # NISP-H (reddest)
        "blue_idx":   3,   # VIS  →  color = VIS - H
        "catalog":    str(ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"),
        "catalog_hdu": 1,
        "color_bins": np.linspace(-0.5, 4.5, 24),
        # real-flux loader: returns (flux_red, flux_blue) in µJy
        "flux_cols":  {"red": "flux_h_templfit", "blue": "flux_vis_psf"},
        "z_col":      "z_lephare",
        "mass_col":   "logM_lephare",
        "sfr_col":    "logSFR_lephare",
        "snr_cols":   ["flux_h_templfit", "fluxerr_h_templfit"],
    },
    "jwst": {
        "names":      ["F115W", "F150W", "F277W", "F444W"],
        "red_idx":    3,   # F444W
        "blue_idx":   2,   # F277W → color = F277W - F444W
        "catalog":    "/home/msiudek/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits",
        "catalog_hdu": (1, 2),   # phot HDU1, ref HDU2
        "color_bins": np.linspace(-1.0, 2.5, 22),
        "stems":      ["f115w", "f150w", "f277w", "f444w"],
        "z_col":      "zpdf_med",
        "mass_col":   "mass_med",
        "sfr_col":    "sfr_med",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", choices=["euclid", "jwst"], required=True)
    p.add_argument("--atlas-name", required=True)
    p.add_argument("--catalog", default=None, help="Override catalog path")
    p.add_argument("--outdir", required=True)
    p.add_argument("--snr-min", type=float, default=3.0)
    return p.parse_args()


def abmag(f_ujy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f_ujy > 0, -2.5 * np.log10(f_ujy / 3631e6), np.nan)


def lum_proxy(flux_ujy, z):
    z = np.clip(z, 1e-4, None)
    d_l_cm = COSMO.luminosity_distance(z).to("cm").value
    return flux_ujy * 1e-29 * 4.0 * np.pi * d_l_cm**2


def load_atlas(atlas_name, cfg):
    data = hickle.load(str(LIB_DIR / atlas_name))
    logM   = np.array(data["mstar"], dtype=float)
    z      = np.array(data["zval"],  dtype=float)
    seds   = np.array(data["sed"],   dtype=float)
    logSFR = np.array(data["sfr"],   dtype=float)
    flux_red  = seds[:, cfg["red_idx"]]
    flux_blue = seds[:, cfg["blue_idx"]]
    return logM, z, flux_red, flux_blue, logSFR


def load_real_euclid(cfg, catalog, snr_min):
    cat = Table.read(catalog)
    fred  = np.array(cat[cfg["flux_cols"]["red"]],  dtype=float)   # µJy
    fblue = np.array(cat[cfg["flux_cols"]["blue"]], dtype=float)
    z    = np.array(cat[cfg["z_col"]],    dtype=float)
    logM = np.array(cat[cfg["mass_col"]], dtype=float)
    sfr  = np.array(cat[cfg["sfr_col"]],  dtype=float)
    # SNR on H band
    ferr_col = cfg["snr_cols"][1]
    if ferr_col in cat.colnames:
        ferr = np.array(cat[ferr_col], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_h = np.abs(fred / np.where(ferr > 0, ferr, np.nan))
    else:
        snr_h = np.full(len(cat), 99.0)
    good = (np.isfinite(z) & (z > 0) & np.isfinite(logM) & (logM > 4)
            & np.isfinite(sfr) & (sfr > -90) & np.isfinite(fred) & (fred > 0)
            & np.isfinite(fblue) & (fblue > 0) & (snr_h >= snr_min))
    return logM[good], z[good], fred[good], fblue[good], sfr[good]


def load_real_jwst(cfg, catalog, snr_min):
    phot = Table.read(catalog, hdu=1)
    ref  = Table.read(catalog, hdu=2)
    stems = cfg["stems"]
    n = len(phot)
    flux = np.zeros((n, 4)); ferr = np.zeros((n, 4))
    for j, stem in enumerate(stems):
        f = np.array(phot[f"flux_aper_{stem}"], dtype=float)
        e = np.array(phot[f"flux_err_aper_{stem}"], dtype=float)
        flux[:, j] = f[:, 0] if f.ndim == 2 else f
        ferr[:, j] = e[:, 0] if e.ndim == 2 else e
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux / np.where(ferr > 0, ferr, np.nan))
    good = np.all(snr >= snr_min, axis=1)
    z    = np.array(ref[cfg["z_col"]],    dtype=float)
    logM = np.array(ref[cfg["mass_col"]], dtype=float)
    sfr  = np.array(ref[cfg["sfr_col"]],  dtype=float)
    good &= (np.isfinite(z) & (z > 0) & np.isfinite(logM) & (logM > 4)
             & np.isfinite(sfr) & (sfr > -90))
    return (logM[good], z[good], flux[good, cfg["red_idx"]],
            flux[good, cfg["blue_idx"]], sfr[good])


def median_offset_by_color(c_a, y_a, c_r, y_r, bins):
    diffs, w = [], []
    for i in range(len(bins) - 1):
        ma = (c_a >= bins[i]) & (c_a < bins[i+1]) & np.isfinite(y_a)
        mr = (c_r >= bins[i]) & (c_r < bins[i+1]) & np.isfinite(y_r)
        if ma.sum() >= 10 and mr.sum() >= 10:
            diffs.append(np.nanmedian(y_a[ma]) - np.nanmedian(y_r[mr]))
            w.append(mr.sum())
    return float(np.average(diffs, weights=w)) if diffs else np.nan


def running_median(c, y, bins, m):
    cc = 0.5 * (bins[:-1] + bins[1:])
    out = []
    for i in range(len(bins) - 1):
        sel = m & (c >= bins[i]) & (c < bins[i+1])
        out.append(np.nanmedian(y[sel]) if sel.sum() >= 10 else np.nan)
    return cc, np.array(out)


def main():
    args = parse_args()
    cfg = SURVEY[args.survey]
    if args.catalog:
        cfg["catalog"] = args.catalog
    color_label = f"{cfg['names'][cfg['blue_idx']]} − {cfg['names'][cfg['red_idx']]}"
    red_name = cfg["names"][cfg["red_idx"]]
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    aM, aZ, aFred, aFblue, aSFR = load_atlas(args.atlas_name, cfg)
    if args.survey == "euclid":
        rM, rZ, rFred, rFblue, rSFR = load_real_euclid(cfg, cfg["catalog"], args.snr_min)
    else:
        rM, rZ, rFred, rFblue, rSFR = load_real_jwst(cfg, cfg["catalog"], args.snr_min)
    print(f"[{args.survey}] atlas N={len(aM)}, real N={len(rM)}")

    a_color = abmag(aFblue) - abmag(aFred)
    r_color = abmag(rFblue) - abmag(rFred)
    aL = lum_proxy(aFred, aZ); rL = lum_proxy(rFred, rZ)
    a_ML, r_ML     = aM - np.log10(aL),   rM - np.log10(rL)
    a_SFRL, r_SFRL = aSFR - np.log10(aL), rSFR - np.log10(rL)

    bins = cfg["color_bins"]
    for quantity, a_y, r_y, ylabel, tag, what in [
        ("M/L",   a_ML,   r_ML,   f"log(M / L_{red_name})",   "ML",   "mass"),
        ("SFR/L", a_SFRL, r_SFRL, f"log(SFR / L_{red_name})", "SFRL", "SFR"),
    ]:
        fig, axes = plt.subplots(1, len(Z_BINS), figsize=(5*len(Z_BINS), 4.5), sharey=True)
        fig.suptitle(f"[{args.survey} v3] {quantity} vs color — ATLAS (blue) vs REAL (red).  "
                     f"atlas above real → {quantity} too high → {what} over-inferred", fontsize=11)
        print(f"\n=== [{args.survey}] {quantity} offset (atlas − real) at matched color ===")
        for ax, (zlo, zhi) in zip(axes, Z_BINS):
            sa = (aZ >= zlo) & (aZ < zhi); sr = (rZ >= zlo) & (rZ < zhi)
            ma = sa & np.isfinite(a_color) & np.isfinite(a_y)
            ax.hexbin(a_color[ma], a_y[ma], gridsize=40, cmap="Blues", mincnt=1, alpha=0.7)
            cc, med_a = running_median(a_color, a_y, bins, ma)
            _,  med_r = running_median(r_color, r_y, bins, sr & np.isfinite(r_color) & np.isfinite(r_y))
            ax.plot(cc, med_a, "b-o", ms=4, lw=2, label="atlas median")
            ax.plot(cc, med_r, "r-s", ms=4, lw=2, label="real median")
            off = median_offset_by_color(a_color[sa], a_y[sa], r_color[sr], r_y[sr], bins)
            ax.text(0.05, 0.95, f"Δ={off:+.2f} dex", transform=ax.transAxes, va="top",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))
            ax.set_title(f"z=[{zlo},{zhi})", fontsize=10)
            ax.set_xlabel(f"{color_label} [mag]"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            print(f"  z=[{zlo},{zhi}): Δlog({quantity}) = {off:+.3f} dex")
        plt.tight_layout()
        out = outdir / f"{args.survey}_{tag}_vs_color_atlas_vs_real.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  ✓ {out.name}")

    print("\nREAD: atlas (blue) ABOVE real (red) in M/L at fixed color → forward model")
    print("assigns too much mass per light → SBI over-infers. Curves overlap → forward")
    print("model calibrated, residual bias is inference/noise not the SEDs.")


if __name__ == "__main__":
    main()
