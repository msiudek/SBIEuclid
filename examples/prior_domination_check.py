"""
Is the Euclid high-z mass bias regression-to-prior-mean?

If, where the SED can't constrain M/L (Euclid z>1, no rest-frame NIR), the
network returns ~the prior's mean mass, then within a z bin the predicted mass
becomes insensitive to the true mass: slope(pred vs true) → 0 and predictions
pile up at a central value. We test that directly from existing inference
results (no model/FSPS needed) and overlay the atlas mass-prior median.

Usage:
    python examples/prior_domination_check.py \
        --inference sbi-logs/inference_euclid_v3/inference_results.npz \
        --atlas library/atlas_euclid_v3_100k_100000_Nparam_2.dbatlas \
        --outdir sbi-logs/prior_domination_euclid_v3
"""

import argparse
from pathlib import Path

import hickle
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Z_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference", required=True)
    p.add_argument("--atlas", default=None, help="atlas .dbatlas to overlay prior-median mass vs z")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    r = np.load(ROOT / args.inference if not Path(args.inference).is_absolute() else args.inference,
                allow_pickle=True)
    mS = r["logM_sbi"].astype(float)
    mT = r["logM_cosmosweb"].astype(float)
    z  = r["z"].astype(float)

    # atlas prior median mass per z bin (the value SBI relaxes toward when uninformative)
    prior_med = {}
    if args.atlas:
        a = hickle.load(str(ROOT / args.atlas if not Path(args.atlas).is_absolute() else args.atlas))
        aM, aZ = np.array(a["mstar"], float), np.array(a["zval"], float)
        for (zlo, zhi) in Z_BINS:
            m = (aZ >= zlo) & (aZ < zhi)
            prior_med[(zlo, zhi)] = np.median(aM[m]) if m.sum() else np.nan

    # ── Panel grid: pred vs true per z bin, with slope ──────────────────
    fig, axes = plt.subplots(1, len(Z_BINS), figsize=(5*len(Z_BINS), 4.8), sharex=True, sharey=True)
    fig.suptitle("Euclid v3: SBI mass vs reference per z bin.  slope→0 & pile-up at prior "
                 "median ⇒ regression to prior (data can't constrain M/L)", fontsize=11)
    print("z-bin        slope   intercept   bias    N   prior_med")
    for ax, (zlo, zhi) in zip(axes, Z_BINS):
        m = (z >= zlo) & (z < zhi) & np.isfinite(mS) & np.isfinite(mT)
        x, y = mT[m], mS[m]
        ax.plot([4, 12], [4, 12], "k--", lw=1)
        ax.scatter(x, y, s=12, alpha=0.45, c="steelblue")
        slope = intercept = np.nan
        if m.sum() >= 10:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, slope*xs + intercept, "r-", lw=2,
                    label=f"slope={slope:.2f}")
        pm = prior_med.get((zlo, zhi), np.nan)
        if np.isfinite(pm):
            ax.axhline(pm, color="green", ls=":", lw=2, label=f"prior med={pm:.1f}")
        ax.set_title(f"z=[{zlo},{zhi})  bias={np.median(y-x):+.2f}", fontsize=10)
        ax.set_xlabel("reference logM (LePhare)")
        ax.set_ylabel("SBI logM")
        ax.legend(fontsize=8); ax.set_xlim(5.5, 12); ax.set_ylim(5.5, 12)
        print(f"[{zlo},{zhi})   {slope:6.2f}   {intercept:7.2f}   {np.median(y-x):+5.2f}  {m.sum():4d}   {pm:6.2f}")
    plt.tight_layout()
    out = outdir / "pred_vs_true_by_z.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n✓ {out}")

    # ── pred & true mass vs z (does SBI compress toward prior median?) ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(z, mT, s=10, alpha=0.4, c="orange", label="reference (LePhare)")
    ax.scatter(z, mS, s=10, alpha=0.4, c="steelblue", label="SBI")
    zc = [0.5*(a+b) for a, b in Z_BINS]
    ax.plot(zc, [np.median(mS[(z>=a)&(z<b)]) for a,b in Z_BINS], "b-o", lw=2, label="SBI median")
    ax.plot(zc, [np.median(mT[(z>=a)&(z<b)]) for a,b in Z_BINS], "-o", color="darkorange", lw=2, label="ref median")
    if prior_med:
        ax.plot(zc, [prior_med[zb] for zb in Z_BINS], "g:s", lw=2, label="atlas prior median")
    ax.set_xlabel("z"); ax.set_ylabel("logM"); ax.legend(fontsize=9)
    ax.set_title("If SBI median tracks the atlas prior median (not the reference) at high z\n"
                 "→ prior-dominated → fix is the mass-prior shape, not the SFH prior", fontsize=10)
    plt.tight_layout()
    out2 = outdir / "mass_vs_z_pile_up.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ {out2}")


if __name__ == "__main__":
    main()
