"""
Empirical M/L calibration as a directional check on the mass-bias diagnosis.

Hypothesis (from the constraint-loss diagnosis): where Euclid loses the
rest-frame-NIR M/L constraint (z>~0.7), the SBI posterior falls back to the
FSPS prior, so the residual

    r = logM_reference - logM_sbi

is NOT random noise but a deterministic function of the *observed* SED shape
(colors) and redshift. If so, a cross-validated regressor trained on
Euclid-only observables (10-band colors + photo-z) should predict r and, when
added back, collapse the bias on held-out galaxies.

CRITICAL: features are Euclid-only observables so the calibration is applicable
to the full survey. The reference mass (LePhare, anchored by deep COSMOS-Web
NIR) is used ONLY as the training target, never as a feature.

Reports out-of-fold (honest, no leakage) bias & NMAD before/after correction,
overall and per redshift bin, and writes a diagnostic figure.

    python examples/ml_calibration.py \
        --npz sbi-logs/inference_euclid_v3/<run>.npz \
        --catalog obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits \
        --ref-key logM_cosmosweb \
        --out sbi-logs/inference_euclid_v3/ml_calibration.png
"""

import argparse
import glob

import numpy as np
from astropy.table import Table
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

# 10 Euclid-pipeline bands actually fed to the SBI model (filters_to_use.dat).
FILTER_STEMS = ["h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
                "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]
Z_BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 6)]


def flux_col(stem, phot_type="templfit"):
    """Match inference_cosmosweb.build_phot_col: VIS uses psf, rest templfit."""
    if phot_type == "templfit":
        return "flux_vis_psf" if stem == "vis" else f"flux_{stem}_templfit"
    return f"flux_{stem}_{phot_type}_aper"


def ab_mag(flux_ujy):
    f = np.asarray(flux_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f / 3631e6), np.nan)


def nmad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="inference npz (or glob dir)")
    p.add_argument("--catalog", required=True)
    p.add_argument("--ref-key", default="logM_cosmosweb",
                   help="reference logM key in npz (default logM_cosmosweb)")
    p.add_argument("--sbi-key", default="logM_sbi")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ml_calibration.png")
    return p.parse_args()


def main():
    args = parse_args()
    npz_path = args.npz
    if not npz_path.endswith(".npz"):
        npz_path = glob.glob(npz_path.rstrip("/") + "/*.npz")[0]
    d = np.load(npz_path)
    logM_sbi = np.asarray(d[args.sbi_key], dtype=float)
    logM_ref = np.asarray(d[args.ref_key], dtype=float)
    z = np.asarray(d["z"], dtype=float)
    sel = np.asarray(d["selected_indices"], dtype=int)
    print(f"loaded {npz_path}: N={len(z)}")

    cat = Table.read(args.catalog)
    mags = np.column_stack([ab_mag(np.asarray(cat[flux_col(s)])[sel])
                            for s in FILTER_STEMS])

    # Features: colors relative to H (reddest Euclid band) + H mag + z.
    # Colors carry M/L shape; H anchors the luminosity scale; z sets rest-frame.
    h = mags[:, 0]
    colors = mags[:, 1:] - h[:, None]            # 9 colors w.r.t. H
    X = np.column_stack([colors, h, z])
    r = logM_ref - logM_sbi                      # residual to predict

    good = np.isfinite(r) & np.isfinite(z) & np.isfinite(h)
    X, r, z, logM_sbi, logM_ref = X[good], r[good], z[good], logM_sbi[good], logM_ref[good]
    print(f"usable (finite ref/H/z): N={good.sum()}")

    # Out-of-fold prediction (no leakage).
    r_hat = np.full_like(r, np.nan)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr, te in kf.split(X):
        m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=args.seed)
        m.fit(X[tr], r[tr])
        r_hat[te] = m.predict(X[te])

    logM_corr = logM_sbi + r_hat
    bias_before = logM_sbi - logM_ref
    bias_after = logM_corr - logM_ref

    def line(name, b):
        print(f"  {name:14s} bias={np.median(b):+.3f}  NMAD={nmad(b):.3f}")

    print("\n=== overall (out-of-fold) ===")
    line("before", bias_before)
    line("after", bias_after)

    print("\n=== bias vs z (median; NMAD) ===")
    print(f"  {'z-bin':12s} {'N':>4s}  {'before':>16s}  {'after':>16s}")
    for lo, hi in Z_BINS:
        mb = (z >= lo) & (z < hi)
        if mb.sum() == 0:
            continue
        print(f"  [{lo},{hi})".ljust(13)
              + f"{mb.sum():>4d}  "
              + f"{np.median(bias_before[mb]):+.3f} ({nmad(bias_before[mb]):.3f})  ".rjust(18)
              + f"{np.median(bias_after[mb]):+.3f} ({nmad(bias_after[mb]):.3f})".rjust(18))

    # ── figure ──────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Label/limits adapt to the calibrated quantity (mass vs SFR).
    qty = "logSFR" if "SFR" in args.ref_key.upper() else "logM"
    allv = np.concatenate([logM_ref, logM_sbi, logM_corr])
    lim = [np.floor(np.nanpercentile(allv, 1)), np.ceil(np.nanpercentile(allv, 99))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (m, ttl) in zip(ax[:2],
                           [(logM_sbi, "before"), (logM_corr, "after (OOF)")]):
        sc = a.scatter(logM_ref, m, c=z, s=12, cmap="viridis", vmin=0, vmax=4)
        a.plot(lim, lim, "k--", lw=1)
        b = m - logM_ref
        a.set(xlim=lim, ylim=lim, xlabel=f"{qty} reference (LePhare)",
              ylabel=f"{qty} SBI", title=f"{ttl}: bias={np.median(b):+.2f} NMAD={nmad(b):.2f}")
        plt.colorbar(sc, ax=a, label="z")
    zc = [0.5 * (lo + hi) for lo, hi in Z_BINS]
    bb = [np.median(bias_before[(z >= lo) & (z < hi)]) for lo, hi in Z_BINS]
    ba = [np.median(bias_after[(z >= lo) & (z < hi)]) for lo, hi in Z_BINS]
    ax[2].plot(zc, bb, "o-", label="before")
    ax[2].plot(zc, ba, "s-", label="after (OOF)")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="z", ylabel=f"median bias ({qty}_sbi - ref)",
              title="bias vs z")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
