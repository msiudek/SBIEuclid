"""
Atlas diagnostics for validating physical trends before training.

Tests implemented:
1) Mass–Magnitude relation (hexbin)
2) Redshift–Magnitude relation (hexbin)
3) Per-band Mass–Flux slopes (log10 flux vs logM)
4) Color vs Redshift relation (hexbin)
5) Flux distribution BEFORE noise (hist, log y)
6) One-galaxy nuisance-fixed mass sweep (shape vs normalization)
7) Narrow-z slope check
8) Control-z regression: logF = a*logM + b*log(1+z) + c
9) Slope vs z-bin
10) Flux vs (mass,z) scatter
11) Tail isolation (flux > threshold)
12) Tail strength per band
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbipix import sbipix
from sbipix.train.simulator import makespec_parametric
from sbipix.utils.sed_utils import (
    flux_ujy_to_mag,
    load_filter_metadata,
    mag_to_flux_ujy,
    sfh_delayed_exponential,
)


ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"


def parse_args():
    parser = argparse.ArgumentParser(description="Run atlas physical diagnostics")
    parser.add_argument("--atlas-name", default="atlas_obs_euclid_north_validate")
    parser.add_argument("--n-sim", type=int, default=2000)
    parser.add_argument("--phot-type", choices=["templfit", "2fwhm", "3fwhm"], default="templfit")
    parser.add_argument("--outdir", default="sbi-logs/atlas_diagnostics")
    parser.add_argument("--mass-mag-band", default="NISP-H", help="Band for tests 1/2/5")
    parser.add_argument("--color-band-1", default="VIS", help="Band1 for color=b1-b2")
    parser.add_argument("--color-band-2", default="NISP-Y", help="Band2 for color=b1-b2")
    return parser.parse_args()


def load_model(args):
    model = sbipix()
    noise_prefix = f"north_{args.phot_type}"
    model.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(OBS_DIR),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    model.atlas_path = str(LIB_DIR) + "/"
    model.atlas_name = args.atlas_name
    model.n_simulation = args.n_sim
    model.parametric = True
    model.both_masses = True
    model.infer_z = False
    model.load_simulation()
    return model


def get_band_index(filter_short, band_name):
    if band_name not in filter_short:
        raise ValueError(f"Band '{band_name}' not found. Available: {filter_short}")
    return filter_short.index(band_name)


def test_mass_mag(logm, mag_band, band_name, outdir):
    valid = np.isfinite(logm) & np.isfinite(mag_band)
    x = logm[valid]
    y = mag_band[valid]
    slope, intercept = np.polyfit(x, y, 1)

    plt.figure(figsize=(6, 5))
    plt.hexbin(x, y, gridsize=50, bins="log", mincnt=1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    plt.plot(xx, slope * xx + intercept, color="crimson", lw=2, label=f"slope={slope:.3f}")
    plt.xlabel("log(M*)")
    plt.ylabel(f"{band_name} mag")
    plt.title("TEST 1: Mass–Magnitude relation")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out = outdir / f"test1_mass_mag_{band_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return slope, out


def test_z_mag(z, mag_band, band_name, outdir):
    valid = np.isfinite(z) & np.isfinite(mag_band)
    x = z[valid]
    y = mag_band[valid]
    slope, intercept = np.polyfit(x, y, 1)

    plt.figure(figsize=(6, 5))
    plt.hexbin(x, y, gridsize=50, bins="log", mincnt=1)
    plt.plot(np.linspace(np.nanmin(x), np.nanmax(x), 200),
             slope * np.linspace(np.nanmin(x), np.nanmax(x), 200) + intercept,
             color="crimson", lw=2, label=f"slope={slope:.3f}")
    plt.xlabel("z")
    plt.ylabel(f"{band_name} mag")
    plt.title("TEST 2: Redshift–Magnitude relation")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out = outdir / f"test2_z_mag_{band_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return slope, out


def test_mass_flux_per_band(logm, obs_mag, filter_short, outdir):
    rows = []
    for i, band in enumerate(filter_short):
        flux = mag_to_flux_ujy(obs_mag[:, i])
        valid = np.isfinite(logm) & np.isfinite(flux) & (flux > 0)
        if np.sum(valid) < 10:
            rows.append((band, np.nan, np.nan, 0))
            continue
        x = logm[valid]
        y = np.log10(flux[valid])
        slope, intercept = np.polyfit(x, y, 1)
        rows.append((band, float(slope), float(intercept), int(np.sum(valid))))

    out_txt = outdir / "test3_mass_flux_slopes_per_band.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("band\tslope\tintercept\tN\n")
        for band, slope, intercept, npt in rows:
            handle.write(f"{band}\t{slope:.6f}\t{intercept:.6f}\t{npt}\n")

    labels = [r[0] for r in rows]
    slopes = [r[1] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.axhline(1.0, color="black", ls="--", lw=1.2, label="expected slope = 1")
    plt.plot(range(len(labels)), slopes, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("slope of log10(flux) vs logM")
    plt.title("TEST 3: Per-band Mass–Flux slopes")
    plt.legend()
    plt.tight_layout()
    out_plot = outdir / "test3_mass_flux_slopes_per_band.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()
    return out_txt, out_plot


def test_color_redshift(z, mag1, mag2, band1, band2, outdir):
    color = mag1 - mag2
    valid = np.isfinite(z) & np.isfinite(color)
    plt.figure(figsize=(6, 5))
    plt.hexbin(z[valid], color[valid], gridsize=50, bins="log", mincnt=1)
    plt.xlabel("z")
    plt.ylabel(f"{band1} - {band2}")
    plt.title("TEST 4: Color vs Redshift")
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out = outdir / f"test4_color_z_{band1.replace('-', '_')}_minus_{band2.replace('-', '_')}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def test_flux_hist(flux_band, band_name, outdir):
    valid = np.isfinite(flux_band) & (flux_band > 0)
    plt.figure(figsize=(6, 4.5))
    plt.hist(flux_band[valid], bins=80, log=True)
    plt.xlabel(f"{band_name} flux (uJy)")
    plt.ylabel("N")
    plt.title("TEST 5: Flux distribution BEFORE noise")
    plt.tight_layout()
    out = outdir / f"test5_flux_hist_{band_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def test_mass_sweep_one_galaxy(model, filter_short, outdir):
    try:
        import fsps
        from astropy.cosmology import FlatLambdaCDM
    except Exception as exc:
        print(f"TEST 6 skipped: fsps unavailable ({exc})")
        return None

    z0 = float(np.nanmedian(model.theta[:, 7]))
    met0 = float(np.nanmedian(model.theta[:, 5]))
    av0 = float(np.nanmedian(model.theta[:, 6]))
    tau0 = float(np.nanmedian(model.theta[:, 3]))
    ti0 = float(np.nanmedian(model.theta[:, 4]))

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    age_gyr = float(cosmo.age(z0).value)
    t = np.linspace(0.001, age_gyr, 1000)

    sp = fsps.StellarPopulation(zcontinuous=1, sfh=3, dust_type=2)
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1

    masses = [8.0, 9.0, 10.0, 11.0]
    mags_all = []

    dense_basis_filter_list = model._prepare_dense_basis_filter_list()

    for logm in masses:
        sfh_gyr, t_axis = sfh_delayed_exponential(t, logmassval=logm, tau=tau0, ti=ti0)
        sfh_yr = np.where(np.isnan(sfh_gyr / 1e9) | (sfh_gyr / 1e9 < 1e-33), 1e-33, sfh_gyr / 1e9)
        sed = makespec_parametric(
            [sfh_yr, t_axis, av0, met0, z0],
            priors=None,
            sp=sp,
            cosmo=cosmo,
            filter_list=dense_basis_filter_list,
            filt_dir=model.filter_path,
            return_spec=False,
            peraa=False,
            input_sfh=True,
        )
        mag = flux_ujy_to_mag(np.asarray(sed, dtype=float))
        mags_all.append(mag)

    mags_all = np.asarray(mags_all)
    x = np.arange(len(filter_short))
    plt.figure(figsize=(10, 4.8))
    for i, logm in enumerate(masses):
        plt.plot(x, mags_all[i], marker='o', lw=1.6, label=f"logM={logm:.1f}")
    plt.gca().invert_yaxis()
    plt.xticks(x, filter_short, rotation=45, ha="right")
    plt.ylabel("AB magnitude")
    plt.title("TEST 6: One-galaxy nuisance-fixed mass sweep")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out = outdir / "test6_one_galaxy_mass_sweep_mags.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def test_mass_flux_narrow_z(logm, obs_mag, z, filter_short, outdir, z_min=0.9, z_max=1.1):
    """
    TEST 7 (A): Per-band mass–flux slopes in narrow redshift slice.
    Expected: slope ~ 0.9–1.0 when cosmology/dust effects suppressed.
    """
    z_mask = (z > z_min) & (z < z_max)
    logm_slice = logm[z_mask]
    obs_mag_slice = obs_mag[z_mask]
    
    n_gal_in_slice = np.sum(z_mask)
    rows = []
    for i, band in enumerate(filter_short):
        flux = mag_to_flux_ujy(obs_mag_slice[:, i])
        valid = np.isfinite(logm_slice) & np.isfinite(flux) & (flux > 0)
        if np.sum(valid) < 5:
            rows.append((band, np.nan, np.nan, 0))
            continue
        x = logm_slice[valid]
        y = np.log10(flux[valid])
        slope, intercept = np.polyfit(x, y, 1)
        rows.append((band, float(slope), float(intercept), int(np.sum(valid))))

    out_txt = outdir / f"test7_mass_flux_narrow_z_{z_min:.1f}_{z_max:.1f}.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write(f"# Narrow z slice: {z_min:.1f} < z < {z_max:.1f}, N_gal={n_gal_in_slice}\n")
        handle.write("band\tslope\tintercept\tN\n")
        for band, slope, intercept, npt in rows:
            handle.write(f"{band}\t{slope:.6f}\t{intercept:.6f}\t{npt}\n")

    labels = [r[0] for r in rows]
    slopes = [r[1] for r in rows]
    plt.figure(figsize=(10, 4))
    plt.axhline(1.0, color="black", ls="--", lw=1.2, label="expected slope = 1")
    plt.plot(range(len(labels)), slopes, marker="o", color="tab:blue")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("slope of log10(flux) vs logM")
    plt.title(f"TEST 7 (A): Per-band slopes in narrow z slice [{z_min:.1f}, {z_max:.1f}]  (N={n_gal_in_slice})")
    plt.legend()
    plt.tight_layout()
    out_plot = outdir / f"test7_mass_flux_narrow_z_{z_min:.1f}_{z_max:.1f}.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()
    
    # Return median slope across bands for summary
    valid_slopes = [s for s in slopes if np.isfinite(s)]
    median_slope = float(np.median(valid_slopes)) if valid_slopes else np.nan
    return median_slope, out_txt, out_plot


def test_mass_flux_control_redshift(logm, z, obs_mag, filter_short, outdir):
    """
    TEST B: Control redshift in regression
    Fit logF = a*logM + b*log(1+z) + c  per band.
    """
    rows = []
    log1pz = np.log10(1.0 + np.maximum(z, 0.0))
    for i, band in enumerate(filter_short):
        flux = mag_to_flux_ujy(obs_mag[:, i])
        valid = (
            np.isfinite(logm)
            & np.isfinite(log1pz)
            & np.isfinite(flux)
            & (flux > 0)
        )
        if np.sum(valid) < 10:
            rows.append((band, np.nan, np.nan, np.nan, 0))
            continue

        y = np.log10(flux[valid])
        X = np.column_stack([logm[valid], log1pz[valid], np.ones(np.sum(valid))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = beta
        rows.append((band, float(a), float(b), float(c), int(np.sum(valid))))

    out_txt = outdir / "test8_mass_flux_control_redshift.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("band\ta_logM\tb_log1pz\tc\tN\n")
        for band, a, b, c, npt in rows:
            handle.write(f"{band}\t{a:.6f}\t{b:.6f}\t{c:.6f}\t{npt}\n")

    labels = [r[0] for r in rows]
    avals = [r[1] for r in rows]
    bvals = [r[2] for r in rows]

    plt.figure(figsize=(10, 4.5))
    plt.axhline(1.0, color="black", ls="--", lw=1.2, label="expected a = 1")
    plt.plot(range(len(labels)), avals, marker="o", label="a (logM coeff)")
    plt.plot(range(len(labels)), bvals, marker="s", label="b (log(1+z) coeff)")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("regression coefficients")
    plt.title("TEST 8 (B): logF = a*logM + b*log(1+z) + c")
    plt.legend()
    plt.tight_layout()
    out_plot = outdir / "test8_mass_flux_control_redshift.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()

    finite_a = [a for a in avals if np.isfinite(a)]
    median_a = float(np.median(finite_a)) if finite_a else np.nan
    return median_a, out_txt, out_plot


def test_mass_flux_slope_vs_z_bins(logm, z, obs_mag, filter_short, outdir):
    """
    TEST C: slope of log10(flux) vs logM in redshift bins.
    Uses the selected mass-mag band for compactness.
    """
    z_bins = [
        (0.0, 1.0, "z<1"),
        (1.0, 2.0, "1<z<2"),
        (2.0, 3.0, "2<z<3"),
        (3.0, 4.0, "3<z<4"),
        (4.0, 6.0, "4<z<6"),
    ]

    rows = []
    for band_idx, band in enumerate(filter_short):
        flux = mag_to_flux_ujy(obs_mag[:, band_idx])
        for z_lo, z_hi, z_label in z_bins:
            z_mask = (z >= z_lo) & (z < z_hi)
            valid = z_mask & np.isfinite(logm) & np.isfinite(flux) & (flux > 0)
            npt = int(np.sum(valid))
            if npt < 10:
                rows.append((band, z_label, np.nan, np.nan, npt))
                continue

            x = logm[valid]
            y = np.log10(flux[valid])
            slope, intercept = np.polyfit(x, y, 1)
            rows.append((band, z_label, float(slope), float(intercept), npt))

    out_txt = outdir / "test9_mass_flux_slope_vs_z_bins.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("band\tz_bin\tslope\tintercept\tN\n")
        for band, z_label, slope, intercept, npt in rows:
            handle.write(f"{band}\t{z_label}\t{slope:.6f}\t{intercept:.6f}\t{npt}\n")

    # Plot only for the first (reference) band to keep visualization readable.
    ref_band = filter_short[0]
    ref_rows = [r for r in rows if r[0] == ref_band]
    labels = [r[1] for r in ref_rows]
    slopes = [r[2] for r in ref_rows]

    plt.figure(figsize=(7, 4.2))
    plt.axhline(1.0, color="black", ls="--", lw=1.2, label="expected slope = 1")
    plt.plot(range(len(labels)), slopes, marker="o")
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(f"slope in {ref_band}")
    plt.title("TEST 9 (C): slope vs z-bin")
    plt.legend()
    plt.tight_layout()
    out_plot = outdir / "test9_mass_flux_slope_vs_z_bins.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()

    return out_txt, out_plot


def test10_flux_vs_mass_z(logm, z, flux_band, band_name, outdir):
    """TEST 10: scatter z vs flux, colored by logM."""
    valid = np.isfinite(logm) & np.isfinite(z) & np.isfinite(flux_band) & (flux_band > 0)
    if np.sum(valid) < 10:
        return None

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(z[valid], flux_band[valid], c=logm[valid], s=10, alpha=0.7, cmap="viridis")
    plt.yscale("log")
    plt.xlabel("z")
    plt.ylabel(f"{band_name} flux (uJy)")
    plt.title("TEST 10: Flux vs redshift (color=logM)")
    cbar = plt.colorbar(sc)
    cbar.set_label("logM")
    plt.tight_layout()
    out = outdir / f"test10_flux_vs_z_colored_logM_{band_name.replace('-', '_')}.png"
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def test11_isolate_tail(logm, z, flux_band, band_name, outdir, threshold=1e3):
    """TEST 11: list tail objects with flux > threshold."""
    valid = np.isfinite(logm) & np.isfinite(z) & np.isfinite(flux_band)
    tail = valid & (flux_band > threshold)

    out = outdir / f"test11_tail_objects_{band_name.replace('-', '_')}.txt"
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(f"# Tail isolation for {band_name}\n")
        handle.write(f"# threshold_uJy={threshold}\n")
        handle.write("idx\tz\tlogM\tflux_uJy\n")
        idx = np.where(tail)[0]
        for i in idx:
            handle.write(f"{i}\t{z[i]:.6f}\t{logm[i]:.6f}\t{flux_band[i]:.6e}\n")

    if np.sum(tail) == 0:
        summary = {
            "n_tail": 0,
            "z_min": np.nan,
            "z_med": np.nan,
            "z_max": np.nan,
            "logm_min": np.nan,
            "logm_med": np.nan,
            "logm_max": np.nan,
        }
    else:
        summary = {
            "n_tail": int(np.sum(tail)),
            "z_min": float(np.nanmin(z[tail])),
            "z_med": float(np.nanmedian(z[tail])),
            "z_max": float(np.nanmax(z[tail])),
            "logm_min": float(np.nanmin(logm[tail])),
            "logm_med": float(np.nanmedian(logm[tail])),
            "logm_max": float(np.nanmax(logm[tail])),
        }
    return summary, out


def test12_tail_band_dependence(obs_mag, filter_short, outdir, threshold=1e3):
    """TEST 12: quantify high-flux tail per band."""
    rows = []
    for i, band in enumerate(filter_short):
        flux = mag_to_flux_ujy(obs_mag[:, i])
        valid = np.isfinite(flux) & (flux > 0)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            rows.append((band, 0, np.nan, np.nan, np.nan))
            continue
        tail = valid & (flux > threshold)
        n_tail = int(np.sum(tail))
        frac_tail = n_tail / n_valid
        p99, p999 = np.percentile(flux[valid], [99, 99.9])
        rows.append((band, n_tail, frac_tail, float(p99), float(p999)))

    out_txt = outdir / "test12_tail_per_band.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write(f"band\tn_tail_gt_{threshold:.0f}uJy\tfrac_tail\tp99_uJy\tp99.9_uJy\n")
        for band, n_tail, frac_tail, p99, p999 in rows:
            handle.write(f"{band}\t{n_tail}\t{frac_tail:.6e}\t{p99:.6e}\t{p999:.6e}\n")

    labels = [r[0] for r in rows]
    fracs = [r[2] for r in rows]
    plt.figure(figsize=(10, 4.3))
    plt.bar(range(len(labels)), fracs)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(f"fraction with flux > {threshold:.0f} uJy")
    plt.title("TEST 12: Tail strength per band")
    plt.tight_layout()
    out_plot = outdir / "test12_tail_per_band.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()

    return out_txt, out_plot


def main():
    args = parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    filter_meta = load_filter_metadata("filters_to_use.dat", filt_dir=str(OBS_DIR))
    filter_short = [m["short"] for m in filter_meta]

    model = load_model(args)
    logm = np.asarray(model.theta[:, 0], dtype=float)
    z = np.asarray(model.theta[:, 7], dtype=float)
    obs_mag = np.asarray(model.obs, dtype=float)

    bi = get_band_index(filter_short, args.mass_mag_band)
    b1 = get_band_index(filter_short, args.color_band_1)
    b2 = get_band_index(filter_short, args.color_band_2)

    slope_mass_mag, out1 = test_mass_mag(logm, obs_mag[:, bi], args.mass_mag_band, outdir)
    slope_z_mag, out2 = test_z_mag(z, obs_mag[:, bi], args.mass_mag_band, outdir)
    out3_txt, out3_plot = test_mass_flux_per_band(logm, obs_mag, filter_short, outdir)
    out4 = test_color_redshift(z, obs_mag[:, b1], obs_mag[:, b2], args.color_band_1, args.color_band_2, outdir)
    out5 = test_flux_hist(mag_to_flux_ujy(obs_mag[:, bi]), args.mass_mag_band, outdir)
    out6 = test_mass_sweep_one_galaxy(model, filter_short, outdir)
    slope_narrow_z, out7_txt, out7_plot = test_mass_flux_narrow_z(logm, obs_mag, z, filter_short, outdir, z_min=0.9, z_max=1.1)
    median_a_ctrlz, out8_txt, out8_plot = test_mass_flux_control_redshift(logm, z, obs_mag, filter_short, outdir)
    out9_txt, out9_plot = test_mass_flux_slope_vs_z_bins(logm, z, obs_mag, filter_short, outdir)

    flux_band = mag_to_flux_ujy(obs_mag[:, bi])
    flux_valid = flux_band[np.isfinite(flux_band) & (flux_band > 0)]
    p99, p999, p9999 = np.percentile(flux_valid, [99, 99.9, 99.99])
    out10 = test10_flux_vs_mass_z(logm, z, flux_band, args.mass_mag_band, outdir)
    tail_summary, out11_txt = test11_isolate_tail(logm, z, flux_band, args.mass_mag_band, outdir, threshold=1e3)
    out12_txt, out12_plot = test12_tail_band_dependence(obs_mag, filter_short, outdir, threshold=1e3)

    print("\nAtlas diagnostics complete")
    print(f"  TEST1 slope (mag vs logM): {slope_mass_mag:.3f} (expected ~ -2.5)")
    print(f"  TEST2 slope (mag vs z): {slope_z_mag:.3f} (expected > 0)")
    print(f"  TEST7A median slope (narrow z ∈ [0.9,1.1]): {slope_narrow_z:.3f} (expected ~ 0.9–1.0)")
    print(f"  TEST8B median a (control-z regression): {median_a_ctrlz:.3f} (expected ~ 1)")
    print(
        f"  Flux percentiles ({args.mass_mag_band}): "
        f"p99={p99:.3e}, p99.9={p999:.3e}, p99.99={p9999:.3e} uJy"
    )
    print(
        f"  TEST11 tail ({args.mass_mag_band}, flux>1e3 uJy): "
        f"N={tail_summary['n_tail']}, "
        f"z[min/med/max]=[{tail_summary['z_min']:.3f}, {tail_summary['z_med']:.3f}, {tail_summary['z_max']:.3f}], "
        f"logM[min/med/max]=[{tail_summary['logm_min']:.3f}, {tail_summary['logm_med']:.3f}, {tail_summary['logm_max']:.3f}]"
    )
    print("  Outputs:")
    for path in [
        out1, out2, out3_txt, out3_plot, out4, out5, out6,
        out7_txt, out7_plot, out8_txt, out8_plot, out9_txt, out9_plot,
        out10, out11_txt, out12_txt, out12_plot
    ]:
        if path is not None:
            print(f"    {path}")


if __name__ == "__main__":
    main()
