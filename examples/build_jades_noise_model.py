"""
Build an INTEGRATED JADES noise model from the catalog flux errors, in the
sbipix consumption format (per-filter σ(mag) in magnitude bins).

Unlike the paper's PIXEL noise model (built from mosaic pixels), this derives
the noise from the integrated catalog photometry, so a model trained with it
sees noise matching integrated galaxies. Output filenames use --prefix so they
load via sbipix with sx.obs_prefix=<prefix>:

    mean_sigma_<prefix>.npy   (nfilt, nbins)   mean σ_mag per filter/bin
    std_sigma_<prefix>.npy    (nfilt, nbins)   std σ_mag per filter/bin
    percentiles_<prefix>.npy  (nperc, nfilt)   mag thresholds per filter
    background_noise_<prefix>.npy (nfilt,)     1σ depth (µJy) per filter

    python examples/build_jades_noise_model.py \
        --catalog obs/obs_properties/JADES/hlsp_jades_..._catalog.fits \
        --filter-list obs/obs_properties/filters_jades_no_wfc.dat \
        --aperture kron_conv --prefix jades_integrated
"""

import argparse
from pathlib import Path

import numpy as np

from jades_catalog import bands_from_filter_list, load_photometry

ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--filter-list", default=str(OBS_DIR / "filters_jades_no_wfc.dat"))
    p.add_argument("--aperture", default="kron_conv",
                   choices=["kron_conv", "kron", "circ_conv", "circ"])
    p.add_argument("--circ-index", type=int, default=2)
    p.add_argument("--prefix", default="jades_integrated")
    p.add_argument("--percentiles", type=float, nargs="*", default=[20, 40, 60, 80],
                   help="mag percentile thresholds defining the bins (default 20 40 60 80 -> 5 bins)")
    p.add_argument("--out-dir", default=str(OBS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    from sbipix.utils.sed_utils import mag_conversion

    bands = bands_from_filter_list(args.filter_list)
    nfilt = len(bands)
    print(f"{nfilt} bands: {bands}")

    flux, err, meta = load_photometry(args.catalog, bands, aperture=args.aperture,
                                      circ_index=args.circ_index)
    # characterize noise on real galaxies (FLAG_ST bitmask: star=1, galaxy=32768)
    gal = meta["flag_st"] != 1
    flux, err = flux[gal], err[gal]
    print(f"using {gal.sum()} galaxies for noise characterization")

    nthr = len(args.percentiles)
    nbins = nthr + 1
    mean_sigma = np.full((nfilt, nbins), np.nan)
    std_sigma = np.full((nfilt, nbins), np.nan)
    percentiles = np.full((nthr, nfilt), np.nan)
    background = np.full(nfilt, np.nan)

    for i in range(nfilt):
        f_i, e_i = flux[:, i], err[:, i]
        ok = np.isfinite(f_i) & (f_i > 0) & np.isfinite(e_i) & (e_i > 0)
        if ok.sum() < 50:
            print(f"  [{bands[i]}] only {ok.sum()} detections — filling defaults")
            background[i] = np.nanmedian(e_i[np.isfinite(e_i) & (e_i > 0)]) if np.any(e_i > 0) else 1e-3
            mean_sigma[i, :] = 1.0
            std_sigma[i, :] = 0.5
            percentiles[:, i] = np.linspace(24, 30, nthr)
            continue
        mag = mag_conversion(f_i[ok], convert_to="mag")
        magerr = np.abs(e_i[ok] * (-2.5 / (np.log(10) * f_i[ok])))
        thr = np.nanpercentile(mag, args.percentiles)
        percentiles[:, i] = thr
        bin_idx = np.digitize(mag, thr)            # 0..nthr
        for b in range(nbins):
            m = bin_idx == b
            if m.sum() > 0:
                mean_sigma[i, b] = np.nanmean(magerr[m])
                std_sigma[i, b] = np.nanstd(magerr[m])
        # fill any empty bin with nearest finite value
        for b in range(nbins):
            if not np.isfinite(mean_sigma[i, b]):
                mean_sigma[i, b] = np.nanmean(mean_sigma[i, :])
                std_sigma[i, b] = np.nanmean(std_sigma[i, :])
        # 1σ depth: median flux error of faint detections (µJy)
        faint = mag > np.nanpercentile(mag, 80)
        background[i] = float(np.nanmedian(e_i[ok][faint])) if faint.sum() else float(np.nanmedian(e_i[ok]))
        print(f"  [{bands[i]:6s}] depth(1σ)={background[i]:.4g} µJy  "
              f"σ_mag bins={np.round(mean_sigma[i], 3)}")

    outd = Path(args.out_dir)
    np.save(outd / f"mean_sigma_{args.prefix}.npy", mean_sigma)
    np.save(outd / f"std_sigma_{args.prefix}.npy", std_sigma)
    np.save(outd / f"percentiles_{args.prefix}.npy", percentiles)
    np.save(outd / f"background_noise_{args.prefix}.npy", background)
    print(f"\n✓ wrote 4 files with prefix '{args.prefix}' to {outd}")
    print(f"  shapes: mean_sigma {mean_sigma.shape}, percentiles {percentiles.shape}, "
          f"background {background.shape}")


if __name__ == "__main__":
    main()
