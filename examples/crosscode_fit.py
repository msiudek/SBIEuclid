"""Cross-code library test: fit CIGALE best-model SEDs with the FSPS atlas.

CIGALE's results.fits stores the best-fit model flux in every band. Those are
NOISELESS synthetic SEDs whose true (CIGALE/BC03) stellar mass is known
exactly. Fitting them with our FSPS atlas at fixed z isolates the pure
library-vs-library mass scale: any offset found here cannot come from
observational noise, selection, emission lines in the data, or priors on
the real sample.

Result (2026-07-08, v3 100k atlas vs Khostovan-matched CIGALE euclid_total):
recovered - truth = +0.240 dex median, z-trend +0.15 -> +0.36 — the FSPS
atlas assigns +0.25 dex more stellar mass than BC03/CIGALE to identical SEDs.

Usage:
    python examples/crosscode_fit.py \
        --atlas library/atlas_euclid_v3_100k_100000_Nparam_2.dbatlas \
        --cigale sbi-logs/cigale_khostovan_matched/euclid_total/results.fits
"""
import argparse

import numpy as np
import hickle
from astropy.io import fits

# CIGALE model-flux columns in the atlas band order
# [h, j, y, vis, g_hsc, z_hsc, g_decam, r_decam, i_decam, z_decam]
CIGALE_COLS = ['best.Euclid_NISP_H', 'best.Euclid_NISP_J', 'best.Euclid_NISP_Y',
               'best.Euclid_VIS', 'best.HSC_g', 'best.HSC_z',
               'best.DECam_g', 'best.DECam_r', 'best.DECam_i', 'best.DECam_z']
ZBINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="library/atlas_euclid_v3_100k_100000_Nparam_2.dbatlas")
    ap.add_argument("--cigale", default="sbi-logs/cigale_khostovan_matched/euclid_total/results.fits")
    ap.add_argument("--real-fits",
                    default="sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits",
                    help="catalog carrying spec-z per object_id (CIGALE results lack z)")
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--dz", type=float, default=0.02)
    ap.add_argument("--frac-err", type=float, default=0.05,
                    help="fractional error assumed for the chi2 weights (SEDs are noiseless)")
    args = ap.parse_args()

    atl = hickle.load(args.atlas)
    seds = np.asarray(atl['sed'], dtype=float)
    zt = np.asarray(atl['zval'], dtype=float)
    mt = np.asarray(atl['mstar'], dtype=float)
    o = np.argsort(zt)
    zt, seds, mt = zt[o], seds[o], mt[o]

    c = fits.open(args.cigale)[1].data
    F = np.stack([np.asarray(c[k], dtype=float) * 1e3 for k in CIGALE_COLS], axis=1)  # mJy->uJy
    m_true = np.log10(np.asarray(c['best.stellar.m_star'], dtype=float))

    r = fits.open(args.real_fits)[1].data
    cid = {int(i): k for k, i in enumerate(c['id'])}
    rows = np.array([cid.get(int(x), -1) for x in r['object_id']])
    sel = rows >= 0
    zg = np.asarray(r['z_lephare'], dtype=float)[sel]
    F, m_true = F[rows[sel]], m_true[rows[sel]]

    rng = np.random.default_rng(0)
    pick = rng.choice(len(F), size=min(args.n_fit, len(F)), replace=False)
    res, zs = [], []
    for g in pick:
        zv = zg[g]
        if not (0.01 < zv < zt.max()):
            continue
        lo = np.searchsorted(zt, zv - args.dz * (1 + zv))
        hi = np.searchsorted(zt, zv + args.dz * (1 + zv))
        if hi - lo < 20:
            continue
        dat = F[g]
        if not np.all(np.isfinite(dat)) or np.any(dat <= 0):
            continue
        T = seds[lo:hi]
        w = 1.0 / (args.frac_err * dat) ** 2
        num = (T * (dat * w)).sum(axis=1)
        den = (T ** 2 * w).sum(axis=1)
        A = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        chi2 = ((dat - A[:, None] * T) ** 2 * w).sum(axis=1)
        chi2[A <= 0] = np.inf
        b = int(np.argmin(chi2))
        res.append(mt[lo:hi][b] + np.log10(A[b]) - m_true[g])
        zs.append(zv)

    res, zs = np.array(res), np.array(zs)
    nmad = 1.4826 * np.median(np.abs(res - np.median(res)))
    print(f"CROSS-CODE: atlas fit of noiseless CIGALE model SEDs (truth = CIGALE best mass), N={len(res)}")
    print(f"  median(recovered - truth) = {np.median(res):+.3f}   NMAD = {nmad:.3f}")
    for zlo, zhi in ZBINS:
        s = (zs >= zlo) & (zs < zhi)
        if s.sum() > 30:
            print(f"    z {zlo}-{zhi}: {np.median(res[s]):+.3f}  n={s.sum()}")


if __name__ == "__main__":
    main()
