"""
Extra, easier-to-read diagnostics for the Khostovan spec-z four-way test.

Figures (written to sbi-logs/cigale_khostovan_specz/):
  fig1_mass_mass.png   : logM(estimator) vs logM(Khostovan LePhare) for
                         pipeline / SBI / CIGALE(Euclid) / CIGALE(full)
  fig2_decomposition.png : SBI bias vs z split into 3 additive pieces
                         SBI-CIG(Euclid)  +  CIG(Euclid)-CIG(full)  +  CIG(full)-LePhare
  fig3_bias_vs_mass.png : median mass bias vs reference logM
  fig4_sfr.png         : logSFR(SBI) and logSFR(CIGALE Euclid) vs Khostovan LePhare SFR
"""
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "sbi-logs" / "cigale_khostovan_specz"
SBI = ROOT / "sbi-logs" / "inference_khostovan_v3"
MDIR = ROOT / "sbi-logs" / "cigale_khostovan_matched"
MATCH = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/matched_andrea_khostovan.fits")


def nmad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def med(x, m):
    v = x[m]; v = v[np.isfinite(v)]
    return np.median(v) if len(v) else np.nan


def load():
    ref = Table.read(D / "cigale_khostovan_reference.csv")
    rmap = {int(i): k for k, i in enumerate(np.asarray(ref["id"]).astype(np.int64))}

    inp = Table.read(D / "sbi_input_khostovan.fits")
    oid_all = np.asarray(inp["object_id"]).astype(np.int64)
    npz = np.load(SBI / "inference_results.npz")
    sel = np.asarray(npz["selected_indices"], dtype=int)
    oid = oid_all[sel]
    sbiM = np.asarray(npz["logM_sbi"], float)
    sbiSFR = np.asarray(npz["logSFR_sbi"], float)
    z = np.asarray(npz["z"], float)

    def cig(path):
        t = Table.read(path)
        i = np.asarray(t["id"]).astype(np.int64)
        M = np.log10(np.asarray(t["bayes.stellar.m_star"], float))
        S = np.log10(np.asarray(t["bayes.sfh.sfr"], float))
        chi = np.asarray(t["best.reduced_chi_square"], float)
        return {int(k): (M[j], S[j], chi[j]) for j, k in enumerate(i)}
    cE = cig(MDIR / "euclid" / "results.fits")
    cF = cig(MDIR / "full" / "results.fits")

    m = Table.read(MATCH)
    lpsfr = {int(i): float(s) for i, s in
             zip(np.asarray(m["object_id"]).astype(np.int64),
                 np.asarray(m["khostovan_lp_sfr_med"], float))}

    rows = []
    for k, o in enumerate(oid):
        r = rmap.get(int(o)); ce = cE.get(int(o)); cf = cF.get(int(o))
        if r is None or ce is None or cf is None:
            continue
        rows.append((z[k], sbiM[k], sbiSFR[k],
                     float(ref["pipeline_logM"][r]),
                     float(ref["khostovan_lp_mass_med"][r]),
                     ce[0], cf[0], ce[1], cf[1], ce[2],
                     lpsfr.get(int(o), np.nan)))
    A = np.array(rows)
    keys = "z sbiM sbiSFR pipe lp cigEM cigFM cigESFR cigFSFR chi lpSFR".split()
    return {k: A[:, j] for j, k in enumerate(keys)}


def main():
    d = load()
    z, sbiM, pipe, lp = d["z"], d["sbiM"], d["pipe"], d["lp"]
    cigEM, cigFM = d["cigEM"], d["cigFM"]
    ok = (np.isfinite(pipe) & np.isfinite(sbiM) & np.isfinite(cigEM) &
          np.isfinite(cigFM) & np.isfinite(lp) & (lp > 4) & (lp < 13) & (d["chi"] < 20))
    print(f"N total {len(z)}, N good(chi2<20) {ok.sum()}")

    # ---------- Fig 1: mass-mass panels vs Khostovan LePhare ----------
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2), sharex=True, sharey=True)
    panels = [("Euclid pipeline\n(Phosphoros)", pipe),
              ("SBI (v3)\nEuclid bands", sbiM),
              ("CIGALE\nEuclid bands", cigEM),
              ("CIGALE\nfull+IRAC", cigFM)]
    for a, (name, y) in zip(ax, panels):
        b = (y - lp)[ok]
        sc = a.scatter(lp[ok], y[ok], c=z[ok], s=6, cmap="viridis", vmin=0, vmax=4)
        a.plot([7, 12], [7, 12], "k--", lw=1)
        a.set(xlim=[7, 12], ylim=[7, 12], xlabel="logM* Khostovan LePhare (full+IRAC)")
        a.set_title(f"{name}\nbias {np.median(b):+.2f}, NMAD {nmad(b):.2f}", fontsize=10)
    ax[0].set_ylabel("logM* estimator")
    fig.colorbar(sc, ax=ax, label="spec-z", fraction=0.02)
    fig.savefig(D / "fig1_mass_mass.png", dpi=110, bbox_inches="tight")
    print("wrote fig1_mass_mass.png")

    # ---------- Fig 2: additive decomposition of SBI bias vs z ----------
    ZB = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 6)]
    zc = [0.5 * (a + b) for a, b in ZB]

    def trend(x):
        return [med(x, ok & (z >= a) & (z < b)) if (ok & (z >= a) & (z < b)).sum() > 8
                else np.nan for a, b in ZB]
    tot = trend(sbiM - lp)          # total SBI bias
    method = trend(sbiM - cigEM)    # SBI vs CIGALE, same Euclid bands
    bandc = trend(cigEM - cigFM)    # band coverage (lost IRAC)
    code = trend(cigFM - lp)        # CIGALE vs LePhare, full bands
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(zc, tot, "ko-", lw=2, label="TOTAL: SBI − LePhare(full)")
    ax[0].plot(zc, method, "s-", color="crimson", label="method: SBI − CIGALE(Euclid)")
    ax[0].plot(zc, bandc, "^-", color="tab:blue", label="band coverage: CIGALE(Euclid−full)")
    ax[0].plot(zc, code, "d-", color="grey", label="code: CIGALE(full) − LePhare")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xlabel="spec-z", ylabel="median Δ logM* [dex]",
              title="SBI mass bias = method + band-coverage + code")
    ax[0].legend(fontsize=8)
    # stacked bar view per z-bin
    x = np.arange(len(ZB))
    ax[1].bar(x, method, label="method (SBI−CIGALE Euclid)", color="crimson")
    ax[1].bar(x, bandc, bottom=method, label="band coverage", color="tab:blue")
    ax[1].bar(x, code, bottom=np.array(method) + np.array(bandc), label="code", color="grey")
    ax[1].plot(x, tot, "ko-", label="total (check)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{a}-{b}" for a, b in ZB], fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="spec-z bin", ylabel="Δ logM* [dex]", title="stacked contributions")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(D / "fig2_decomposition.png", dpi=120)
    print("wrote fig2_decomposition.png")

    # ---------- Fig 3: bias vs reference mass ----------
    MB = np.arange(8, 11.6, 0.5)
    mc = 0.5 * (MB[:-1] + MB[1:])

    def trend_m(x):
        out = []
        for i in range(len(MB) - 1):
            mm = ok & (lp >= MB[i]) & (lp < MB[i + 1])
            out.append(med(x, mm) if mm.sum() > 8 else np.nan)
        return out
    fig, a = plt.subplots(figsize=(7.5, 5.5))
    a.plot(mc, trend_m(pipe - lp), "o-", label="pipeline − LePhare")
    a.plot(mc, trend_m(sbiM - lp), "s-", color="crimson", label="SBI − LePhare")
    a.plot(mc, trend_m(cigEM - lp), "^-", color="tab:green", label="CIGALE(Euclid) − LePhare")
    a.plot(mc, trend_m(cigEM - cigFM), "d--", color="tab:blue", label="band coverage")
    a.axhline(0, color="k", lw=0.6)
    a.set(xlabel="logM* Khostovan LePhare", ylabel="median Δ logM* [dex]",
          title="mass bias vs reference mass")
    a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(D / "fig3_bias_vs_mass.png", dpi=120)
    print("wrote fig3_bias_vs_mass.png")

    # ---------- Fig 4: SFR comparison ----------
    lpSFR = d["lpSFR"]; sbiSFR = d["sbiSFR"]; cigESFR = d["cigESFR"]
    oksfr = ok & np.isfinite(lpSFR) & (lpSFR > -90) & np.isfinite(sbiSFR) & np.isfinite(cigESFR)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.3))
    for a, (name, y) in zip(ax[:2], [("SBI", sbiSFR), ("CIGALE(Euclid)", cigESFR)]):
        b = (y - lpSFR)[oksfr]
        sc = a.scatter(lpSFR[oksfr], y[oksfr], c=z[oksfr], s=6, cmap="plasma", vmin=0, vmax=4)
        a.plot([-2, 4], [-2, 4], "k--", lw=1)
        a.set(xlim=[-2, 4], ylim=[-2, 4], xlabel="logSFR Khostovan LePhare",
              ylabel=f"logSFR {name}", title=f"{name}: bias {np.median(b):+.2f}, NMAD {nmad(b):.2f}")
    plt.colorbar(sc, ax=ax[1], label="spec-z")
    # SFR bias vs z
    def trend_s(x):
        return [med(x, oksfr & (z >= a) & (z < b)) if (oksfr & (z >= a) & (z < b)).sum() > 8
                else np.nan for a, b in ZB]
    ax[2].plot(zc, trend_s(sbiSFR - lpSFR), "s-", color="crimson", label="SBI − LePhare")
    ax[2].plot(zc, trend_s(cigESFR - lpSFR), "^-", color="tab:green", label="CIGALE(Euclid) − LePhare")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="spec-z", ylabel="median Δ logSFR [dex]", title="SFR bias vs z")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(D / "fig4_sfr.png", dpi=120)
    print("wrote fig4_sfr.png")

    # console summary
    print("\n=== overall (chi2<20) ===")
    print(f"  SBI-LePhare       {med(sbiM-lp,ok):+.3f}")
    print(f"  = method(SBI-CIGe) {med(sbiM-cigEM,ok):+.3f}")
    print(f"  + bandcov(CIGe-CIGf){med(cigEM-cigFM,ok):+.3f}")
    print(f"  + code(CIGf-LePh)  {med(cigFM-lp,ok):+.3f}")
    print(f"  SFR: SBI-LePhare {med(sbiSFR-lpSFR,oksfr):+.3f}  CIGe-LePhare {med(cigESFR-lpSFR,oksfr):+.3f}  (N={oksfr.sum()})")


if __name__ == "__main__":
    main()
