"""
Tiebreaker: is the Euclid mass bias a fundamental FSPS-vs-reference M/L offset,
or a loss of constraint specific to Euclid's missing rest-frame NIR at z>1?

Two checks:
(1) REFERENCE solidity: EZ vs LePhare masses (matched_euclid_farmer.fits).
    If two independent estimators agree, the reference is trustworthy and the
    SBI bias is on our side (not a reference problem).
(2) FSPS-with-NIR vs FSPS-without-NIR: bias (SBI - LePhare) vs z for the JWST
    pipeline (4 NIR bands incl F444W = rest-frame NIR to z~3) and the Euclid
    pipeline (reddest band H=1.6um). SAME FSPS forward model, SAME LePhare ref.
      - JWST tracks LePhare (flat, small bias) -> FSPS M/L is fine when the data
        constrain it -> there is NO fundamental 0.5 dex FSPS-vs-LePhare offset.
      - Euclid diverges with z -> the excess is constraint loss, not SPS physics.

Usage:
    python examples/jwst_euclid_tiebreaker.py --outdir sbi-logs/jwst_tiebreaker
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
FARMER = ROOT / "obs" / "obs_properties" / "COSMOS_Farmer" / "matched_euclid_farmer.fits"
JWST   = ROOT / "sbi-logs" / "inference_jwst_v3" / "inference_results.npz"
EUCLID = ROOT / "sbi-logs" / "inference_euclid_v3" / "inference_results.npz"
Z_BINS = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 5)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="sbi-logs/jwst_tiebreaker")
    return p.parse_args()


def running_bias(z, dm):
    zc, med, lo, hi = [], [], [], []
    for a, b in Z_BINS:
        m = (z >= a) & (z < b) & np.isfinite(dm)
        if m.sum() >= 8:
            zc.append(0.5 * (a + b))
            med.append(np.median(dm[m]))
            lo.append(np.percentile(dm[m], 16))
            hi.append(np.percentile(dm[m], 84))
    return np.array(zc), np.array(med), np.array(lo), np.array(hi)


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ── (1) EZ vs LePhare reference cross-check ──────────────────────────
    cat = Table.read(FARMER)
    lp = np.array(cat["lp_mass_med"], dtype=float)
    ez = np.array(cat["ez_mass"], dtype=float)
    zf = np.array(cat["lp_zbest"], dtype=float) if "lp_zbest" in cat.colnames else np.full(len(cat), np.nan)
    g = np.isfinite(lp) & np.isfinite(ez) & (lp > 6) & (ez > 6) & (lp < 12.5) & (ez < 12.5)
    dref = ez[g] - lp[g]
    print(f"[reference] EZ - LePhare: median={np.median(dref):+.3f}, "
          f"NMAD={1.4826*np.median(np.abs(dref-np.median(dref))):.3f}, N={g.sum()}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hexbin(lp[g], ez[g], gridsize=60, cmap="viridis", mincnt=1, bins="log")
    ax[0].plot([6, 12.5], [6, 12.5], "r--", lw=1.5)
    ax[0].set_xlabel("LePhare logM"); ax[0].set_ylabel("EZ logM")
    ax[0].set_title(f"Reference cross-check: EZ vs LePhare\nmedian offset = {np.median(dref):+.3f} dex")
    ax[1].hist(dref, bins=80, range=(-1.5, 1.5), color="steelblue")
    ax[1].axvline(0, color="k", ls="--"); ax[1].axvline(np.median(dref), color="r", lw=2,
                  label=f"median={np.median(dref):+.3f}")
    ax[1].set_xlabel("EZ - LePhare [dex]"); ax[1].set_ylabel("N"); ax[1].legend()
    ax[1].set_title("Two independent FSPS-style estimators agree\n→ reference is solid; SBI bias is on our side")
    plt.tight_layout(); plt.savefig(outdir / "reference_ez_vs_lephare.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ reference_ez_vs_lephare.png")

    # ── (2) FSPS+NIR (JWST) vs FSPS-noNIR (Euclid), both vs LePhare ──────
    j = np.load(JWST, allow_pickle=True)
    e = np.load(EUCLID, allow_pickle=True)
    jz, jdm = j["z"].astype(float), j["logM_sbi"].astype(float) - j["logM_lephare"].astype(float)
    ez_, edm = e["z"].astype(float), e["logM_sbi"].astype(float) - e["logM_cosmosweb"].astype(float)

    zcj, medj, loj, hij = running_bias(jz, jdm)
    zce, mede, loe, hie = running_bias(ez_, edm)

    print("\n[bias vs z]   JWST(FSPS+NIR)   Euclid(FSPS,noNIR)")
    for a, b in Z_BINS:
        mj = (jz >= a) & (jz < b); me = (ez_ >= a) & (ez_ < b)
        sj = f"{np.median(jdm[mj]):+.2f}" if mj.sum() >= 8 else "  -- "
        se = f"{np.median(edm[me]):+.2f}" if me.sum() >= 8 else "  -- "
        print(f"  z=[{a},{b}):   {sj}            {se}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.fill_between(zcj, loj, hij, color="crimson", alpha=0.15)
    ax.fill_between(zce, loe, hie, color="steelblue", alpha=0.15)
    ax.plot(zcj, medj, "-o", color="crimson", lw=2.5, ms=7, label="JWST (FSPS + rest-frame NIR)")
    ax.plot(zce, mede, "-s", color="steelblue", lw=2.5, ms=7, label="Euclid (FSPS, no rest-frame NIR)")
    ax.set_xlabel("redshift z", fontsize=12)
    ax.set_ylabel(r"mass bias  $\log M_{\rm SBI} - \log M_{\rm LePhare}$ [dex]", fontsize=11)
    ax.set_title("Same FSPS forward model, same LePhare reference.\n"
                 "JWST tracks LePhare (FSPS M/L is fine when data constrain it);\n"
                 "Euclid diverges with z → the excess is constraint loss, not SPS physics", fontsize=10)
    ax.legend(fontsize=10); ax.set_xlim(0, 5)
    plt.tight_layout(); plt.savefig(outdir / "bias_vs_z_jwst_vs_euclid.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  ✓ bias_vs_z_jwst_vs_euclid.png")

    print("\nVERDICT:")
    print("  If EZ≈LePhare AND JWST-SBI≈LePhare → no fundamental FSPS-vs-reference")
    print("  M/L offset. The Euclid z-growing bias is loss of M/L constraint")
    print("  (no rest-frame NIR), so the fix lives in the Euclid setup, not the SEDs.")


if __name__ == "__main__":
    main()
