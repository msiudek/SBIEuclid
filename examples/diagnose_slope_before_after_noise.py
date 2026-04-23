"""
diagnose_slope_before_after_noise.py

Compare slope(logF vs logM | z-bin) BEFORE and AFTER noise injection
to identify where the mass–flux relation collapses.

Usage:
    python examples/diagnose_slope_before_after_noise.py \
        --atlas-name atlas_obs_euclid_north_validate_50000_Nparam_2.dbatlas \
        --n-sim 50000 --phot-type templfit

Output: printed table + sbi-logs/slope_diagnostic/slope_before_after_noise.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbipix import sbipix

OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"

NONDET_MAG = 99.0

Z_BINS = [
    (0.00, 0.25, "z=[0.00,0.25)"),
    (0.25, 0.50, "z=[0.25,0.50)"),
    (0.50, 1.00, "z=[0.50,1.00)"),
    (1.00, 1.50, "z=[1.00,1.50)"),
    (1.50, 2.00, "z=[1.50,2.00)"),
    (2.00, 3.00, "z=[2.00,3.00)"),
    (3.00, 5.00, "z=[3.00,5.00)"),
]

BAND_NAMES = [
    "NISP-H", "NISP-J", "NISP-Y", "VIS",
    "HSC-g", "HSC-z", "DECam-g", "DECam-r", "DECam-i", "DECam-z",
]


def mag_to_logflux(mag):
    """Convert AB mag to log10(flux/uJy), ignoring non-detections."""
    with np.errstate(invalid="ignore"):
        flux = 3631e6 * 10 ** (-0.4 * mag)
    return np.log10(np.maximum(flux, 1e-30))


def flux_to_logflux(flux):
    """Convert flux/uJy to log10(flux), ignoring non-positive values."""
    flux = np.asarray(flux, dtype=float)
    return np.where(flux > 0, np.log10(np.maximum(flux, 1e-30)), np.nan)


def flux_to_asinh(flux, softening):
    """Sign-preserving transform for flux that keeps negative noisy realizations."""
    flux = np.asarray(flux, dtype=float)
    s = max(float(softening), 1e-12)
    return np.arcsinh(flux / s)


def slope_logflux_vs_logm(logm, logflux, min_n=20):
    """OLS slope of logflux vs logM; returns nan if too few points."""
    ok = np.isfinite(logm) & np.isfinite(logflux)
    if ok.sum() < min_n:
        return np.nan, int(ok.sum())
    x = logm[ok]
    y = logflux[ok]
    slope = np.polyfit(x, y, 1)[0]
    return float(slope), int(ok.sum())


def slope_mag_vs_logm(logm, mag, min_n=20):
    """OLS slope of mag vs logM (non-detections excluded); returns nan if too few."""
    det = (mag < NONDET_MAG - 0.5) & np.isfinite(mag) & np.isfinite(logm)
    if det.sum() < min_n:
        return np.nan, int(det.sum())
    x = logm[det]
    y = mag[det]
    slope = np.polyfit(x, y, 1)[0]
    return float(slope), int(det.sum())


def run_diagnostics(sx, logm, z, obs_raw, obs_noisy, filter_names, outdir, observation_space, limits):
    n_bands = obs_raw.shape[1]

    rows = []
    # Reference band: NISP-H (index 0) used for the summary, all bands stored
    for bi in range(n_bands):
        band = filter_names[bi] if bi < len(filter_names) else f"band{bi}"

        before_mag = obs_raw[:, bi]
        after_obs = obs_noisy[:, bi, 0]

        before_logflux = mag_to_logflux(before_mag)
        if observation_space == "flux":
            softening = limits[bi] if bi < len(limits) else np.nanmedian(limits)
            before_flux = 3631e6 * 10 ** (-0.4 * before_mag)
            before_logflux = flux_to_asinh(before_flux, softening)
            after_logflux_det = flux_to_asinh(after_obs, softening)
            after_mag = np.where(after_obs > 0, -2.5 * np.log10(after_obs / 3631e6), NONDET_MAG)
            det_mask = np.isfinite(after_obs)
        else:
            after_mag = after_obs
            after_logflux_det = np.where(after_mag < NONDET_MAG - 0.5,
                                         mag_to_logflux(after_mag), np.nan)
            det_mask = after_mag < NONDET_MAG - 0.5

        for z_lo, z_hi, z_label in Z_BINS:
            zmask = (z >= z_lo) & (z < z_hi)
            n_z = int(zmask.sum())

            # BEFORE noise: slope on raw atlas (all objects, logflux space)
            s_before, n_before = slope_logflux_vs_logm(logm[zmask], before_logflux[zmask])

            # AFTER noise: slope on detected (logflux space)
            s_after_lf, n_after_lf = slope_logflux_vs_logm(logm[zmask], after_logflux_det[zmask])

            # AFTER noise: slope on detected (mag space, as sanity check)
            s_after_mag, n_after_mag = slope_mag_vs_logm(logm[zmask], after_mag[zmask])

            ratio_lf = s_after_lf / s_before if (np.isfinite(s_before) and np.isfinite(s_after_lf)
                                                   and abs(s_before) > 1e-6) else np.nan

            # Detection fraction in this z-bin/band
            if n_z > 0:
                det_frac = float(np.sum(det_mask[zmask])) / n_z
            else:
                det_frac = np.nan

            rows.append({
                "band": band,
                "z_label": z_label,
                "z_lo": z_lo,
                "z_hi": z_hi,
                "n_z": n_z,
                "slope_before_lf": s_before,
                "n_before": n_before,
                "slope_after_lf": s_after_lf,
                "n_after_lf": n_after_lf,
                "slope_after_mag": s_after_mag,
                "n_after_mag": n_after_mag,
                "ratio_lf": ratio_lf,
                "det_frac": det_frac,
            })

    # ── Print summary table ──
    header = (
        f"{'band':<10} {'z_bin':<18} {'N_z':>6} "
        f"{'slope_before':>13} {'slope_after_lf':>15} {'ratio_lf':>9} "
        f"{'slope_after_mag':>16} {'det_frac':>9}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    if observation_space == "flux":
        print("  SLOPE DIAGNOSTIC: asinh(flux/softening) vs logM | z-bin  (BEFORE and AFTER noise)")
    else:
        print("  SLOPE DIAGNOSTIC: log10(flux) vs logM | z-bin  (BEFORE and AFTER noise)")
    print(sep)
    print(header)
    print(sep)

    collapse_count = 0
    total_count = 0

    for r in rows:
        if not np.isfinite(r["slope_before_lf"]) and not np.isfinite(r["slope_after_lf"]):
            continue
        ratio_str = f"{r['ratio_lf']:+.3f}" if np.isfinite(r["ratio_lf"]) else "   nan"
        det_str = f"{r['det_frac']:.2f}" if np.isfinite(r["det_frac"]) else "  nan"
        sb = f"{r['slope_before_lf']:+.4f}" if np.isfinite(r["slope_before_lf"]) else "     nan"
        sa = f"{r['slope_after_lf']:+.4f}" if np.isfinite(r["slope_after_lf"]) else "     nan"
        sam = f"{r['slope_after_mag']:+.4f}" if np.isfinite(r["slope_after_mag"]) else "     nan"
        flag = " *** COLLAPSE" if (np.isfinite(r["ratio_lf"]) and r["ratio_lf"] < 0.5) else ""
        print(
            f"{r['band']:<10} {r['z_label']:<18} {r['n_z']:>6} "
            f"{sb:>13} {sa:>15} {ratio_str:>9} "
            f"{sam:>16} {det_str:>9}{flag}"
        )
        if np.isfinite(r["ratio_lf"]):
            total_count += 1
            if r["ratio_lf"] < 0.5:
                collapse_count += 1

    print(sep)
    if total_count > 0:
        print(
            f"  Collapse summary: {collapse_count}/{total_count} (band,z) bins with ratio_lf < 0.5  "
            f"({100*collapse_count/total_count:.1f}%)"
        )

    # ── Save to file ──
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / "slope_before_after_noise.txt"
    with open(out_txt, "w") as fh:
        fh.write("band\tz_label\tn_z\tslope_before_lf\tn_before\t"
                 "slope_after_lf\tn_after_lf\tslope_after_mag\tn_after_mag\t"
                 "ratio_lf\tdet_frac\n")
        for r in rows:
            fh.write(
                f"{r['band']}\t{r['z_label']}\t{r['n_z']}\t"
                f"{r['slope_before_lf']:.6f}\t{r['n_before']}\t"
                f"{r['slope_after_lf']:.6f}\t{r['n_after_lf']}\t"
                f"{r['slope_after_mag']:.6f}\t{r['n_after_mag']}\t"
                f"{r['ratio_lf']:.6f}\t{r['det_frac']:.4f}\n"
            )
    print(f"\n  Results saved to: {out_txt}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-name", default="atlas_obs_euclid_north_validate_50000_Nparam_2.dbatlas")
    p.add_argument("--n-sim", type=int, default=50000)
    p.add_argument("--phot-type", choices=["templfit", "2fwhm", "3fwhm"], default="templfit")
    p.add_argument("--outdir", default="sbi-logs/slope_diagnostic")
    p.add_argument("--sigma-sampler", default="mag_lognormal",
                   choices=["mag_lognormal", "empirical", "truncnorm"])
    p.add_argument("--detection-model", default="hard", choices=["hard", "probabilistic"])
    p.add_argument("--observation-space", default="mag", choices=["mag", "flux"])
    p.add_argument("--snr-threshold", type=float, default=3.0)
    args = p.parse_args()

    outdir = ROOT / args.outdir
    noise_prefix = f"north_{args.phot_type}"

    # Normalise atlas name (strip .dbatlas suffix if present)
    atlas_stem = args.atlas_name
    if atlas_stem.endswith(".dbatlas"):
        atlas_stem = atlas_stem[:-len(".dbatlas")]
    import re
    atlas_stem = re.sub(r"_\d+_Nparam_\d+$", "", atlas_stem)

    print(f"Atlas stem: {atlas_stem}")
    print(f"N_sim: {args.n_sim}")
    print(f"Phot type: {args.phot_type}  (noise prefix: {noise_prefix})")
    print(f"Sigma sampler: {args.sigma_sampler}  |  Detection model: {args.detection_model}  "
            f"|  Observation space: {args.observation_space}  |  SNR threshold: {args.snr_threshold}")

    sx = sbipix()
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(OBS_DIR),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    sx.atlas_path = str(LIB_DIR) + "/"
    sx.atlas_name = atlas_stem
    sx.n_simulation = args.n_sim
    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False
    sx.include_limit = True
    sx.include_sigma = True
    sx.condition_sigma = True
    sx.configure_noise_model(
        sigma_sampler=args.sigma_sampler,
        detection_model=args.detection_model,
        observation_space=args.observation_space,
    )
    sx.snr_threshold = args.snr_threshold

    print("\n[1/3] Loading atlas simulation...")
    sx.load_simulation()
    sx.load_obs_features()

    logm = np.asarray(sx.theta[:, 0], dtype=float)
    z    = np.asarray(sx.theta[:, 7], dtype=float)
    obs_raw = np.asarray(sx.obs, dtype=float)

    print(f"  Loaded N={len(logm)} galaxies, {obs_raw.shape[1]} bands")
    print(f"  logM range: [{np.nanmin(logm):.2f}, {np.nanmax(logm):.2f}]")
    print(f"  z range:    [{np.nanmin(z):.3f}, {np.nanmax(z):.3f}]")

    print("\n[2/3] Injecting noise...")
    sx.add_noise_nan_limit_all()
    obs_noisy = np.asarray(sx.mag, dtype=float)  # (n, n_filt, 2)

    # Quick global stats
    if args.observation_space == "flux":
        det_frac_global = float(np.mean(np.isfinite(obs_noisy[:, :, 0])))
    else:
        det_frac_global = float(np.mean(obs_noisy[:, :, 0] < NONDET_MAG - 0.5))
    print(f"  Global detection fraction (all bands): {det_frac_global:.3f}")

    print("\n[3/3] Computing slopes...")
    rows = run_diagnostics(
        sx,
        logm,
        z,
        obs_raw,
        obs_noisy,
        BAND_NAMES,
        outdir,
        args.observation_space,
        np.asarray(sx.limits, dtype=float),
    )

    # ── Compact NISP-H only summary ──
    print("\n  NISP-H only summary (ratio_lf = slope_after / slope_before in logflux space):")
    h_rows = [r for r in rows if r["band"] == "NISP-H"]
    for r in h_rows:
        ratio_str = f"{r['ratio_lf']:+.3f}" if np.isfinite(r["ratio_lf"]) else "  nan"
        det_str = f"{r['det_frac']:.2f}" if np.isfinite(r["det_frac"]) else " nan"
        flag = " <-- COLLAPSE" if (np.isfinite(r["ratio_lf"]) and r["ratio_lf"] < 0.5) else ""
        print(f"    {r['z_label']}: ratio={ratio_str}  det_frac={det_str}{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
