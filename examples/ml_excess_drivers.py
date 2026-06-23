"""
Which prior knob drives the atlas M/L excess?

For each atlas galaxy: residual = M/L_atlas − median(M/L_real) at its (color, z).
Then correlate that residual against the atlas's own physical parameters
(metallicity, dust Av, τ, t_i, sSFR, hidden-mass fraction mformed−mstar).
The parameter with the strongest correlation is the lever to bend so the atlas
M/L lands on the real relation.

Euclid-focused (the science goal). Real = matched_euclid_cosmosweb.fits (LePhare).

Usage:
    python examples/ml_excess_drivers.py \
        --atlas-name atlas_euclid_v3_100k_100000_Nparam_2.dbatlas \
        --outdir sbi-logs/ml_excess_drivers_euclid_v3
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.stats import spearmanr

ROOT    = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "library"
COSMO   = FlatLambdaCDM(H0=70, Om0=0.3)

CATALOG = str(ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits")
RED_IDX, BLUE_IDX = 0, 3            # NISP-H, VIS  (color = VIS - H)
COLOR_BINS = np.linspace(-0.5, 4.5, 26)
Z_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]   # reliable VIS-H regime (z<3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-name", required=True)
    p.add_argument("--catalog", default=CATALOG)
    p.add_argument("--outdir", required=True)
    p.add_argument("--snr-min", type=float, default=3.0)
    return p.parse_args()


def abmag(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f / 3631e6), np.nan)


def lum_proxy(f, z):
    z = np.clip(z, 1e-4, None)
    d = COSMO.luminosity_distance(z).to("cm").value
    return f * 1e-29 * 4 * np.pi * d**2


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- atlas ----
    a = hickle.load(str(LIB_DIR / args.atlas_name))
    sfh = np.array(a["sfh_tuple"])          # [mstar, mformed, sfr, tau, ti, Nparam]
    aM   = np.array(a["mstar"], dtype=float)
    aZ   = np.array(a["zval"],  dtype=float)
    aSED = np.array(a["sed"],   dtype=float)
    aSFR = np.array(a["sfr"],   dtype=float)
    aMet = np.array(a["met"],   dtype=float)
    aDust= np.array(a["dust"],  dtype=float)
    aMformed = sfh[:, 1]
    aTau = sfh[:, 3]
    aTi  = sfh[:, 4]

    a_color = abmag(aSED[:, BLUE_IDX]) - abmag(aSED[:, RED_IDX])
    a_ML = aM - np.log10(lum_proxy(aSED[:, RED_IDX], aZ))

    # ---- real ----
    cat = Table.read(args.catalog)
    rFred  = np.array(cat["flux_h_templfit"], dtype=float)
    rFblue = np.array(cat["flux_vis_psf"],    dtype=float)
    rZ = np.array(cat["z_lephare"],   dtype=float)
    rM = np.array(cat["logM_lephare"], dtype=float)
    ferr = np.array(cat["fluxerr_h_templfit"], dtype=float) if "fluxerr_h_templfit" in cat.colnames else None
    snr = np.abs(rFred / np.where((ferr is not None) & (ferr > 0), ferr, np.nan)) if ferr is not None else np.full(len(cat), 99.0)
    good = (np.isfinite(rZ) & (rZ > 0) & np.isfinite(rM) & (rM > 4)
            & np.isfinite(rFred) & (rFred > 0) & np.isfinite(rFblue) & (rFblue > 0)
            & (snr >= args.snr_min))
    rZ, rM, rFred, rFblue = rZ[good], rM[good], rFred[good], rFblue[good]
    r_color = abmag(rFblue) - abmag(rFred)
    r_ML = rM - np.log10(lum_proxy(rFred, rZ))

    print(f"atlas N={len(aM)}, real N={len(rM)}")

    # ---- per-atlas residual = M/L_atlas - median(M/L_real) at (color,z) ----
    drivers = {
        "metallicity [M/H]": aMet,
        "dust Av":           aDust,
        "log10(tau/Gyr)":    np.log10(np.clip(aTau, 1e-3, None)),
        "log10(ti/Gyr)":     np.log10(np.clip(aTi, 1e-3, None)),
        "log sSFR":          aSFR - aM,
        "hidden mass (mformed-mstar)": aMformed - aM,
    }

    summary = {}
    for (zlo, zhi) in Z_BINS:
        sa = (aZ >= zlo) & (aZ < zhi)
        sr = (rZ >= zlo) & (rZ < zhi)
        # real median M/L per color bin in this z slice
        real_med = np.full(len(COLOR_BINS) - 1, np.nan)
        for i in range(len(COLOR_BINS) - 1):
            mr = sr & (r_color >= COLOR_BINS[i]) & (r_color < COLOR_BINS[i+1]) & np.isfinite(r_ML)
            if mr.sum() >= 10:
                real_med[i] = np.nanmedian(r_ML[mr])
        # assign each atlas galaxy the real median at its color bin
        ci = np.digitize(a_color, COLOR_BINS) - 1
        valid = sa & (ci >= 0) & (ci < len(real_med)) & np.isfinite(a_ML)
        resid = np.full(len(aM), np.nan)
        resid[valid] = a_ML[valid] - real_med[np.clip(ci[valid], 0, len(real_med)-1)]
        resid[valid & ~np.isfinite(real_med[np.clip(ci, 0, len(real_med)-1)])] = np.nan

        good_resid = np.isfinite(resid)
        print(f"\n=== z=[{zlo},{zhi}): median M/L excess = {np.nanmedian(resid[good_resid]):+.3f} dex "
              f"(N={good_resid.sum()}) ===")

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"[euclid v3] z=[{zlo},{zhi}): atlas M/L excess vs atlas parameters\n"
                     f"strongest |Spearman r| = the knob driving the M/L over-prediction", fontsize=12)
        rs = {}
        for ax, (name, vals) in zip(axes.flat, drivers.items()):
            m = good_resid & np.isfinite(vals)
            if m.sum() < 50:
                ax.set_visible(False); continue
            r, _ = spearmanr(vals[m], resid[m])
            rs[name] = r
            ax.hexbin(vals[m], resid[m], gridsize=40, cmap="viridis", mincnt=1, bins="log")
            # running median
            qs = np.linspace(np.nanpercentile(vals[m], 2), np.nanpercentile(vals[m], 98), 12)
            cc = 0.5 * (qs[:-1] + qs[1:])
            rm = [np.nanmedian(resid[m & (vals >= qs[i]) & (vals < qs[i+1])])
                  if (m & (vals >= qs[i]) & (vals < qs[i+1])).sum() >= 10 else np.nan
                  for i in range(len(qs)-1)]
            ax.plot(cc, rm, "r-o", ms=4, lw=2)
            ax.axhline(0, color="k", lw=1, ls="--")
            ax.set_xlabel(name); ax.set_ylabel("M/L excess (atlas − real) [dex]")
            ax.set_title(f"{name}:  Spearman r = {r:+.3f}", fontsize=10,
                         fontweight="bold" if abs(r) > 0.3 else "normal")
        plt.tight_layout()
        out = outdir / f"ml_excess_drivers_z{zlo:.0f}_{zhi:.0f}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
        print(f"  ✓ {out}")
        summary[(zlo, zhi)] = rs

    # ---- ranked summary ----
    print("\n" + "=" * 64)
    print("SPEARMAN |r| of M/L-excess vs each knob (higher = stronger lever)")
    print("=" * 64)
    print(f"{'knob':<32}" + "".join(f"z{lo:.0f}-{hi:.0f}".rjust(10) for lo, hi in Z_BINS))
    knobs = list(drivers.keys())
    for k in knobs:
        row = f"{k:<32}"
        for zb in Z_BINS:
            r = summary[zb].get(k, np.nan)
            row += f"{r:+.3f}".rjust(10)
        print(row)
    print("=" * 64)
    print("The knob with consistently largest |r| AND a prior we can shift is the fix.")


if __name__ == "__main__":
    main()
