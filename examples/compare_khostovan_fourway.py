"""
Apples-to-apples stellar-mass comparison on the Khostovan spec-z sample.

Four Euclid-band (no-IRAC) estimates vs full-band IRAC-constrained references,
all on IDENTICAL galaxies at the SAME spec-z:

  Euclid pipeline (Phosphoros)  : pipeline_logM        [reference CSV]
  SBI (euclid_v3)               : inference_results.npz
  CIGALE on Euclid 10 bands     : results.fits (bayes.stellar.m_star)
  reference LePhare (full+IRAC) : khostovan_lp_mass_med [reference CSV]
  reference CIGALE (full+IRAC)  : khostovan_cig_logM    [reference CSV]

Key decompositions:
  method-at-fixed-bands : SBI, CIGALE(Euclid), pipeline all vs LePhare
  pure band coverage    : CIGALE(Euclid) - CIGALE(full)  (same code & priors)
"""
from pathlib import Path
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "sbi-logs" / "cigale_khostovan_specz"      # reference CSV + SBI input
SBI = ROOT / "sbi-logs" / "inference_khostovan_v3"
MDIR = ROOT / "sbi-logs" / "cigale_khostovan_matched"  # Khostovan-matched grid runs
CIG_EUCLID = MDIR / "euclid" / "results.fits"          # CIGALE, 10 Euclid bands
CIG_FULL = MDIR / "full" / "results.fits"              # CIGALE, broad+IRAC (our run)

UJY = 1.0  # input fluxes are µJy


def nmad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def main():
    ref = Table.read(D / "cigale_khostovan_reference.csv")
    ref_id = np.asarray(ref["id"]).astype(np.int64)
    rmap = {int(i): k for k, i in enumerate(ref_id)}

    # --- SBI: map selected rows back to object_id via the input catalog ---
    inp = Table.read(D / "sbi_input_khostovan.fits")
    inp_oid = np.asarray(inp["object_id"]).astype(np.int64)
    npz = np.load(SBI / "inference_results.npz")
    sel = np.asarray(npz["selected_indices"], dtype=int)
    sbi_oid = inp_oid[sel]
    logM_sbi_arr = np.asarray(npz["logM_sbi"], dtype=float)
    # H-mag for the mag-trend plot (flux_h_templfit, µJy)
    hflux = np.asarray(inp["flux_h_templfit"], dtype=float)[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        Hmag = -2.5 * np.log10(hflux / 3631e6)

    # --- CIGALE on Euclid bands (Khostovan-matched grid) ---
    if not CIG_EUCLID.exists():
        raise SystemExit(
            f"Missing {CIG_EUCLID}\nRun the matched-grid Euclid CIGALE first:\n"
            f"  cd {MDIR/'euclid'} && pcigale-filters add filters/*.dat && "
            f"pcigale genconf && pcigale run\n"
            f"(the old sbi-logs/cigale_khostovan_specz/results.fits is a DIFFERENT grid "
            f"-- E_BV_factor=0.44, no dust emission -- and is NOT apples-to-apples with full/)")
    cig = Table.read(CIG_EUCLID)
    cig_oid = np.asarray(cig["id"]).astype(np.int64)
    cig_logM = np.log10(np.asarray(cig["bayes.stellar.m_star"], dtype=float))
    cchi = np.asarray(cig["best.reduced_chi_square"], dtype=float) if \
        "best.reduced_chi_square" in cig.colnames else np.zeros(len(cig))
    cmap = {int(i): (cig_logM[k], cchi[k]) for k, i in enumerate(cig_oid)}

    # --- CIGALE on broad+IRAC (our full-band run, SAME grid) ---
    cigf = Table.read(CIG_FULL)
    fmap = {int(i): np.log10(m) for i, m in
            zip(np.asarray(cigf["id"]).astype(np.int64),
                np.asarray(cigf["bayes.stellar.m_star"], dtype=float))}

    # --- assemble on the SBI galaxies (they are the common inference set) ---
    rows = []
    for k, oid in enumerate(sbi_oid):
        r = rmap.get(int(oid))
        c = cmap.get(int(oid))
        if r is None or c is None:
            continue
        rows.append((
            float(ref["specz"][r]), float(Hmag[k]),
            float(ref["pipeline_logM"][r]),
            float(logM_sbi_arr[k]),
            float(c[0]), float(c[1]),
            float(ref["khostovan_lp_mass_med"][r]),
            float(ref["khostovan_cig_logM"][r]),
            float(fmap.get(int(oid), np.nan)),
        ))
    A = np.array(rows)
    z, Hm, pipe, sbi, cigE, cchi2, lp, cigF_pub, cigF = A.T
    print(f"common galaxies (SBI ∩ CIGALE ∩ ref): {len(A)}")

    ok = np.isfinite(pipe) & np.isfinite(sbi) & np.isfinite(cigE) & np.isfinite(lp) \
        & (cchi2 < 20) & (lp > 4) & (lp < 13)
    print(f"good (finite all, chi2<20, 4<lp<13): {int(ok.sum())}")

    # biases vs LePhare (full+IRAC) reference
    bp, bs, bc = pipe - lp, sbi - lp, cigE - lp
    band = cigE - cigF          # pure band coverage, SAME matched grid both sides
    band_pub = cigE - cigF_pub  # vs Khostovan published full-band CIGALE (66M grid)
    for name, b in [("pipeline", bp), ("SBI", bs), ("CIGALE(Euclid)", bc)]:
        v = b[ok]
        print(f"  {name:16s} bias vs LePhare: {np.median(v):+.3f}  NMAD {nmad(v):.3f}")
    okb = ok & np.isfinite(band)
    print(f"  CIGALE band-coverage (Euclid-full, OUR matched grid): "
          f"{np.median(band[okb]):+.3f}  NMAD {nmad(band[okb]):.3f}  (N={int(okb.sum())})")
    okp = ok & np.isfinite(band_pub)
    print(f"  CIGALE band-coverage (Euclid-Khostovan published):    "
          f"{np.median(band_pub[okp]):+.3f}  (N={int(okp.sum())})")

    # ---- plots ----
    ZB = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 6)]
    zc = [0.5 * (a + b) for a, b in ZB]

    def trend(b, m):
        return [np.median(b[m & (z >= a) & (z < c)]) if (m & (z >= a) & (z < c)).sum() > 5
                else np.nan for a, c in ZB]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    ax[0].plot(zc, trend(bp, ok), "o-", label="pipeline − LePhare")
    ax[0].plot(zc, trend(bs, ok), "s-", label="SBI − LePhare")
    ax[0].plot(zc, trend(bc, ok), "^-", label="CIGALE(Euclid) − LePhare")
    ax[0].plot(zc, trend(band, okb), "d--", color="grey", label="CIGALE(Euclid) − CIGALE(full), same grid")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="spec-z", ylabel="median Δ logM* [dex]",
              title=f"Khostovan spec-z (N={int(ok.sum())}): mass bias vs z")
    ax[0].legend(fontsize=8)

    # bias vs H-mag
    MB = np.arange(20, 26.5, 0.75)
    mc = 0.5 * (MB[:-1] + MB[1:])

    def trend_m(b, m):
        return [np.median(b[m & (Hm >= MB[i]) & (Hm < MB[i + 1])])
                if (m & (Hm >= MB[i]) & (Hm < MB[i + 1])).sum() > 5 else np.nan
                for i in range(len(MB) - 1)]

    ax[1].plot(mc, trend_m(bp, ok), "o-", label="pipeline")
    ax[1].plot(mc, trend_m(bs, ok), "s-", label="SBI")
    ax[1].plot(mc, trend_m(bc, ok), "^-", label="CIGALE(Euclid)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="H mag (templfit)", ylabel="median Δ logM* [dex]",
              title="mass bias vs H magnitude")
    ax[1].legend(fontsize=8)

    # scatter SBI vs LePhare colored by z
    sc = ax[2].scatter(lp[ok], sbi[ok], c=z[ok], s=8, cmap="viridis", vmin=0, vmax=4)
    ax[2].plot([6, 12], [6, 12], "k--", lw=1)
    ax[2].set(xlim=[6, 12], ylim=[6, 12], xlabel="logM LePhare (full+IRAC)",
              ylabel="logM SBI (Euclid bands)",
              title=f"SBI: bias {np.median(bs[ok]):+.2f}, NMAD {nmad(bs[ok]):.2f}")
    plt.colorbar(sc, ax=ax[2], label="spec-z")
    fig.tight_layout()
    out = D / "khostovan_fourway.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    np.savez(D / "khostovan_fourway.npz", z=z, Hmag=Hm, pipeline=pipe, sbi=sbi,
             cigale_euclid=cigE, lephare=lp, cigale_full=cigF,
             cigale_full_published=cigF_pub, ok=ok)


if __name__ == "__main__":
    main()
