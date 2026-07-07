"""Real-vs-mock manifold diagnostic.

Where does real Euclid photometry sit relative to the mock training manifold?
Compares per-band color loci (log10 flux_band/flux_VIS) between the Khostovan
spec-z sample (total-scaled fluxes, as fed to inference) and the mock training
features (from train_euclid.py --dump-features), in redshift bins, with mocks
reweighted to the real VIS-magnitude distribution per z-bin so depth/selection
differences don't masquerade as color offsets.

A systematic real-minus-mock color offset is the off-manifold direction that
drives the training-realization mass zero-point lottery; it is the target of
the per-band zero-point / error-floor calibration to be applied INSIDE the
simulator (template-error-function analog).

Usage:
    python examples/diagnose_manifold.py \
        --mock-npz sbi-logs/train_features_dump.npz \
        --real-fits sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits \
        --out sbi-logs/fig_manifold_diagnostic.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

# band order in filters_to_use.dat / the mock feature matrix
STEMS = ["h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
         "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]
LABELS = ["H", "J", "Y", "VIS", "HSC-g", "HSC-z",
          "DECam-g", "DECam-r", "DECam-i", "DECam-z"]
IVIS = 3
SNR_MIN = 3.0
ZBINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0)]


def flux_to_mag(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, -2.5 * np.log10(f * 1e-6 / 3631.0), np.nan)


def load_real(path):
    """Total-scaled fluxes exactly as inference --phot-type total builds them."""
    t = fits.open(path)[1].data
    scale = t["flux_detection_total"] / t["flux_vis_2fwhm_aper"]
    flux = np.full((len(t), len(STEMS)), np.nan)
    err = np.full_like(flux, np.nan)
    for i, s in enumerate(STEMS):
        flux[:, i] = t[f"flux_{s}_2fwhm_aper"] * scale
        err[:, i] = t[f"fluxerr_{s}_2fwhm_aper"] * np.abs(scale)
    z = np.asarray(t["z_lephare"], dtype=float)
    return flux, err, z


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock-npz", default="sbi-logs/train_features_dump.npz")
    ap.add_argument("--real-fits",
                    default="sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits")
    ap.add_argument("--out", default="sbi-logs/fig_manifold_diagnostic.png")
    args = ap.parse_args()

    d = np.load(args.mock_npz)
    mflux = d["mag"][:, :, 0].astype(float)   # noisy mock flux [uJy]
    msig = d["mag"][:, :, 1].astype(float)
    mz = d["theta"][:, 7].astype(float)

    rflux, rerr, rz = load_real(args.real_fits)
    print(f"mocks: {len(mflux)}   real: {len(rflux)}")

    mvis_mag = flux_to_mag(mflux[:, IVIS])
    rvis_mag = flux_to_mag(rflux[:, IVIS])

    nb = len(STEMS)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    print("\nreal - mock median color offset  Delta log10(f_band/f_VIS)  [dex]")
    print("(mocks reweighted to real VIS-mag distribution per z-bin; "
          f"both require SNR>{SNR_MIN:.0f} in band and VIS)")
    header = "z-bin      " + "".join(f"{l:>9s}" for l in LABELS if l != "VIS")
    print(header)

    for ax, (zlo, zhi) in zip(axes.ravel(), ZBINS):
        rs = (rz >= zlo) & (rz < zhi)
        ms = (mz >= zlo) & (mz < zhi)
        # VIS-mag reweighting of mocks to real distribution
        bins = np.arange(17, 27.01, 0.25)
        rh, _ = np.histogram(rvis_mag[rs], bins=bins)
        mh, _ = np.histogram(mvis_mag[ms], bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mh > 0, rh / np.maximum(mh, 1), 0.0)
        midx = np.clip(np.digitize(mvis_mag[ms], bins) - 1, 0, len(ratio) - 1)
        mw = ratio[midx]
        mw[~np.isfinite(mvis_mag[ms])] = 0.0

        offsets, x = [], []
        row = f"z {zlo:.1f}-{zhi:.1f}  "
        for i in range(nb):
            if i == IVIS:
                continue
            r_ok = rs.copy()
            r_ok[rs] = (rflux[rs, i] > SNR_MIN * rerr[rs, i]) & \
                       (rflux[rs, IVIS] > SNR_MIN * rerr[rs, IVIS])
            m_ok = (mflux[ms, i] > SNR_MIN * msig[ms, i]) & \
                   (mflux[ms, IVIS] > SNR_MIN * msig[ms, IVIS]) & (mw > 0)
            if r_ok.sum() < 50 or m_ok.sum() < 50:
                row += f"{'--':>9s}"
                continue
            rcol = np.log10(rflux[r_ok, i] / rflux[r_ok, IVIS])
            mcol = np.log10(mflux[ms][m_ok, i] / mflux[ms][m_ok, IVIS])
            off = np.median(rcol) - weighted_median(mcol, mw[m_ok])
            offsets.append(off)
            x.append(i)
            row += f"{off:+9.3f}"
        print(row)
        ax.axhline(0, color="k", lw=0.6)
        ax.bar(range(len(offsets)), offsets, color="tab:red", alpha=0.75)
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels([LABELS[i] for i in x], rotation=45, fontsize=8)
        ax.set_title(f"z = {zlo}-{zhi}")
        ax.set_ylim(-0.25, 0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"real $-$ mock  $\Delta\log_{10}(f_b/f_{VIS})$ [dex]")
    fig.suptitle("Off-manifold diagnostic: real Khostovan colors vs mock training colors\n"
                 "(mocks VIS-mag-reweighted per z-bin; positive = real galaxies "
                 "brighter in band relative to VIS than any mock)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
