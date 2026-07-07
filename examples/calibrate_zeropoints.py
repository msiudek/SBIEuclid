"""Per-band zero-point / template-error calibration from fixed-z SED fits.

The analog of LePhare/Phosphoros AUTO_ADAPT: fit every real galaxy at its
spec-z with the noiseless atlas SEDs (free amplitude only), then measure the
per-band median flux residual (obs - model)/model. A residual that is the
same at all z is a photometric zero-point the simulator should apply; the
residual scatter in excess of the photometric errors is the per-band
systematic error floor. Both belong INSIDE the simulator (forward-model
calibration) - no masses or labels are transferred.

Usage:
    python examples/calibrate_zeropoints.py \
        --atlas library/atlas_euclid_v3_100k_100000_Nparam_2.dbatlas \
        --real-fits sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits \
        --out sbi-logs/zeropoints_khostovan.npz
"""
import argparse

import numpy as np
import hickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

STEMS = ["h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
         "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]
LABELS = ["H", "J", "Y", "VIS", "HSC-g", "HSC-z",
          "DECam-g", "DECam-r", "DECam-i", "DECam-z"]
ZBINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]
ERR_FLOOR_FRAC = 0.05   # stabilizer for the chi2 fit only (not applied to results)


def load_real(path):
    t = fits.open(path)[1].data
    scale = t["flux_detection_total"] / t["flux_vis_2fwhm_aper"]
    flux = np.stack([t[f"flux_{s}_2fwhm_aper"] * scale for s in STEMS], axis=1)
    err = np.stack([t[f"fluxerr_{s}_2fwhm_aper"] * np.abs(scale) for s in STEMS], axis=1)
    z = np.asarray(t["z_lephare"], dtype=float)
    logm_ref = np.asarray(t["logM_lephare"], dtype=float)
    return flux.astype(float), err.astype(float), z, logm_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="library/atlas_euclid_v3_100k_100000_Nparam_2.dbatlas")
    ap.add_argument("--real-fits",
                    default="sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits")
    ap.add_argument("--dz", type=float, default=0.02, help="|z_atlas - z_spec| < dz*(1+z)")
    ap.add_argument("--snr-min", type=float, default=3.0)
    ap.add_argument("--nbands-min", type=int, default=6)
    ap.add_argument("--out", default="sbi-logs/zeropoints_khostovan.npz")
    args = ap.parse_args()

    atl = hickle.load(args.atlas)
    seds = np.asarray(atl["sed"], dtype=float)          # (n_templ, 10), noiseless
    zt = np.asarray(atl["zval"], dtype=float)
    mt = np.asarray(atl["mstar"], dtype=float)          # log M* of each template
    order = np.argsort(zt)
    zt, seds, mt = zt[order], seds[order], mt[order]
    # winning-template properties (same sort) for the prior diagnosis
    t_sfr = np.asarray(atl["sfr"], dtype=float)[order]
    t_dust = np.asarray(atl["dust"], dtype=float)[order]
    t_met = np.asarray(atl["met"], dtype=float)[order]
    t_sfh = np.asarray(atl["sfh_tuple"], dtype=float)[order]

    flux, err, z, logm_ref = load_real(args.real_fits)
    n_gal, nb = flux.shape
    logm_fit = np.full(n_gal, np.nan)
    best_idx = np.full(n_gal, -1, dtype=int)   # index into the z-sorted atlas
    sel = (z > 0) & (z < zt.max())
    print(f"real galaxies: {n_gal}, in atlas z-range: {sel.sum()}")

    resid = np.full((n_gal, nb), np.nan)
    chi2n = np.full(n_gal, np.nan)
    n_fit = 0
    idx_gal = np.where(sel)[0]
    for g in idx_gal:
        lo = np.searchsorted(zt, z[g] - args.dz * (1 + z[g]))
        hi = np.searchsorted(zt, z[g] + args.dz * (1 + z[g]))
        if hi - lo < 20:
            continue
        T = seds[lo:hi]                                  # (m, 10)
        d, s = flux[g], err[g]
        valid = np.isfinite(d) & np.isfinite(s) & (s > 0) & (d > args.snr_min * s)
        if valid.sum() < args.nbands_min:
            continue
        s_fit = np.sqrt(s**2 + (ERR_FLOOR_FRAC * np.abs(d))**2)
        w = np.zeros(nb)
        w[valid] = 1.0 / s_fit[valid] ** 2
        # best amplitude per template, then chi2 per template
        num = (T * (d * w)).sum(axis=1)
        den = (T**2 * w).sum(axis=1)
        A = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        chi2 = ((d - A[:, None] * T) ** 2 * w).sum(axis=1)
        chi2[A <= 0] = np.inf
        b = np.argmin(chi2)
        model = A[b] * T[b]
        r = np.where(valid & (model > 0), (d - model) / model, np.nan)
        resid[g] = r
        chi2n[g] = chi2[b] / max(valid.sum() - 1, 1)
        logm_fit[g] = mt[lo:hi][b] + np.log10(A[b])
        best_idx[g] = lo + b
        n_fit += 1

    print(f"fitted: {n_fit} galaxies (dz={args.dz}, >={args.nbands_min} bands SNR>{args.snr_min})")
    ok = np.isfinite(chi2n) & (chi2n < 10)
    print(f"kept for calibration (chi2/nu<10): {ok.sum()}")

    print("\nper-band median residual (obs-model)/model in dex-equivalent "
          "log10(1+r), and NMAD [dex]:")
    print("z-bin      " + "".join(f"{l:>9s}" for l in LABELS))
    dex = np.log10(np.clip(1.0 + resid, 1e-3, None))
    zp = np.full(nb, np.nan)
    zp_sig = np.full(nb, np.nan)
    for zlo, zhi in [(0.0, 5.0)] + ZBINS:
        s = ok & (z >= zlo) & (z < zhi)
        row = f"z {zlo:.1f}-{zhi:.1f}  "
        for i in range(nb):
            v = dex[s, i]
            v = v[np.isfinite(v)]
            if len(v) < 50:
                row += f"{'--':>9s}"
                continue
            med = np.median(v)
            row += f"{med:+9.3f}"
            if zlo == 0.0 and zhi == 5.0:
                zp[i] = med
                zp_sig[i] = 1.4826 * np.median(np.abs(v - med))
        print(row + ("   <- GLOBAL (candidate zero-points)" if (zlo, zhi) == (0.0, 5.0) else ""))
    print("\nGLOBAL NMAD per band [dex]: " +
          " ".join(f"{LABELS[i]}:{zp_sig[i]:.3f}" for i in range(nb)))

    # Template-fit mass check: chi2 best-fit of OUR OWN atlas at fixed spec-z
    # ("LePhare with the FSPS atlas"). If this is unbiased while the flow is
    # +0.3, the atlas SEDs/masses are fine and the offset is the inference
    # machinery (prior weighting); if this is also +0.3, the atlas physics
    # itself carries the bias.
    s = ok & np.isfinite(logm_fit) & np.isfinite(logm_ref)
    dm = logm_fit[s] - logm_ref[s]
    r_corr = np.corrcoef(logm_fit[s], logm_ref[s])[0, 1]
    print(f"\nTEMPLATE-FIT MASS (best-fit atlas SED, free amplitude) vs LePhare, N={s.sum()}:")
    print(f"  median = {np.median(dm):+.3f}  NMAD = {1.4826*np.median(np.abs(dm-np.median(dm))):.3f}  r = {r_corr:.3f}")
    for zlo, zhi in ZBINS:
        b_ = s & (z >= zlo) & (z < zhi)
        if b_.sum() > 50:
            print(f"    z {zlo:.1f}-{zhi:.1f}: {np.median(logm_fit[b_]-logm_ref[b_]):+.3f}  n={b_.sum()}")

    np.savez(args.out, zp_dex=zp, zp_nmad_dex=zp_sig, resid=resid, chi2n=chi2n,
             z=z, labels=np.array(LABELS), logm_fit=logm_fit, logm_ref=logm_ref,
             best_idx=best_idx, atlas_z=zt, atlas_mstar=mt, atlas_sfr=t_sfr,
             atlas_dust=t_dust, atlas_met=t_met, atlas_sfh=t_sfh)
    print(f"saved {args.out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for zlo, zhi in ZBINS:
        s = ok & (z >= zlo) & (z < zhi)
        med = [np.nanmedian(dex[s, i]) for i in range(nb)]
        ax.plot(range(nb), med, "-o", ms=4, label=f"z {zlo}-{zhi}")
    ax.plot(range(nb), zp, "k-s", lw=2, ms=6, label="global")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(nb))
    ax.set_xticklabels(LABELS, rotation=45)
    ax.set_ylabel(r"median $\log_{10}(f_{obs}/f_{model})$ at fixed spec-z [dex]")
    ax.set_title("Fixed-z best-fit atlas residuals (LePhare-style zero-point adaptation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out.replace(".npz", ".png"), dpi=130)
    print(f"saved {args.out.replace('.npz', '.png')}")


if __name__ == "__main__":
    main()
