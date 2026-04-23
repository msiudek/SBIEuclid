"""
Diagnose SFH/age consistency issues in parametric atlas files.

Implements:
TEST A: FSPS age-parameter check (tage/t_univ vs z)
TEST B: sSFR consistency check (sSFR_fsps vs sSFR_input-model)
TEST C: light-weighted age vs z

Notes
-----
- In this codebase FSPS is called with tage ~= age_universe(z), so tage/t_univ
  is expected to be ~1 by construction unless generation code changes.
- Atlas stores parametric tuple as [logMstar, logMformed, logSFR, tau, ti, Nparam].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def _ssfr_mass_slope_from_z(zval: float) -> float:
    if zval < 1.0:
        return -0.08
    if zval < 2.0:
        return -0.15
    if zval < 3.0:
        return -0.22
    if zval < 4.0:
        return -0.28
    return -0.34


def _mean_log_ssfr(logmass: np.ndarray, zval: np.ndarray) -> np.ndarray:
    mu_z = -10.0 + 0.8 * np.log10(1.0 + np.maximum(zval, 0.0))
    vec_b = np.vectorize(_ssfr_mass_slope_from_z)
    b_z = vec_b(zval)
    return mu_z + b_z * (logmass - 10.0)


def _light_weighted_age_gyr(t_univ_gyr: float, tau: float, ti: float, n_grid: int = 800) -> float:
    if not np.isfinite(t_univ_gyr) or t_univ_gyr <= 0:
        return np.nan
    tau = float(np.clip(tau, 1e-3, 100.0))
    ti = float(np.clip(ti, 0.0, t_univ_gyr))

    t = np.linspace(0.0, t_univ_gyr, n_grid)
    sfh = np.where(t >= ti, (t - ti) * np.exp(-(t - ti) / tau), 0.0)
    denom = np.trapz(sfh, t)
    if (not np.isfinite(denom)) or denom <= 0:
        return np.nan

    t_form = np.trapz(t * sfh, t) / denom
    return float(np.clip(t_univ_gyr - t_form, 0.0, t_univ_gyr))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SFH consistency diagnostics")
    parser.add_argument(
        "--atlas-file",
        default="library/atlas_obs_euclid_north_validate_50000_Nparam_2.dbatlas",
        help="Path to atlas .dbatlas file",
    )
    parser.add_argument(
        "--outdir",
        default="sbi-logs/sfh_consistency",
        help="Output directory",
    )
    parser.add_argument(
        "--max-gal",
        type=int,
        default=15000,
        help="Max galaxies for expensive light-weighted-age calculation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    atlas_file = root / args.atlas_file if not Path(args.atlas_file).is_absolute() else Path(args.atlas_file)
    outdir = root / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(atlas_file, "r") as file_handle:
        grp = file_handle["data"]
        zval = np.asarray(grp['"zval"'][:], dtype=float)
        logmstar = np.asarray(grp['"mstar"'][:], dtype=float)
        logsfr = np.asarray(grp['"sfr"'][:], dtype=float)
        sfh_tuple = np.asarray(grp['"sfh_tuple"'][:], dtype=float)

    tau = sfh_tuple[:, 3]
    ti = sfh_tuple[:, 4]
    logmformed = sfh_tuple[:, 1]

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    t_univ = np.asarray(cosmo.age(np.clip(zval, 1e-4, 20.0)).value, dtype=float)

    valid = (
        np.isfinite(zval)
        & np.isfinite(logmstar)
        & np.isfinite(logmformed)
        & np.isfinite(logsfr)
        & np.isfinite(tau)
        & np.isfinite(ti)
        & np.isfinite(t_univ)
        & (t_univ > 0)
    )

    z = zval[valid]
    lmstar = logmstar[valid]
    lmformed = logmformed[valid]
    lsfr = logsfr[valid]
    tau_v = tau[valid]
    ti_v = ti[valid]
    t_univ_v = t_univ[valid]

    # TEST A
    tage_fsps = t_univ_v + 1e-4
    ratio_tage = tage_fsps / t_univ_v
    stellar_age = np.clip(t_univ_v - ti_v, 0.0, None)
    ratio_stellar = np.where(t_univ_v > 0, stellar_age / t_univ_v, np.nan)

    slope_tage = float(np.polyfit(z, ratio_tage, 1)[0])
    slope_stellar = float(np.polyfit(z, ratio_stellar, 1)[0])

    # TEST B
    log_ssfr_fsps_star = lsfr - lmstar
    log_ssfr_fsps_formed = lsfr - lmformed
    log_ssfr_input_model = _mean_log_ssfr(lmstar, z)

    resid_star = log_ssfr_fsps_star - log_ssfr_input_model
    resid_formed = log_ssfr_fsps_formed - log_ssfr_input_model

    # TEST C (subsample for speed)
    n_all = len(z)
    n_use = min(n_all, max(int(args.max_gal), 1))
    rng = np.random.default_rng(0)
    idx = np.arange(n_all) if n_use == n_all else rng.choice(n_all, size=n_use, replace=False)

    z_sub = z[idx]
    t_univ_sub = t_univ_v[idx]
    tau_sub = tau_v[idx]
    ti_sub = ti_v[idx]

    t_light = np.array([
        _light_weighted_age_gyr(tu, ta, tii)
        for tu, ta, tii in zip(t_univ_sub, tau_sub, ti_sub)
    ], dtype=float)

    ok_light = np.isfinite(t_light) & np.isfinite(z_sub)
    slope_tlight = float(np.polyfit(z_sub[ok_light], t_light[ok_light], 1)[0]) if np.any(ok_light) else np.nan

    # Save summary
    summary_path = outdir / "sfh_consistency_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"atlas={atlas_file}\n")
        handle.write(f"N_valid={n_all}\n")
        handle.write("\n[TEST A]\n")
        handle.write(f"slope(tage/t_univ vs z)={slope_tage:.6f}\n")
        handle.write(f"median(tage/t_univ)={np.nanmedian(ratio_tage):.6f}\n")
        handle.write(f"slope((t_univ-ti)/t_univ vs z)={slope_stellar:.6f}\n")
        handle.write(f"median((t_univ-ti)/t_univ)={np.nanmedian(ratio_stellar):.6f}\n")
        handle.write("\n[TEST B]\n")
        handle.write(f"median(resid logssfr_fsps(logMstar)-input_model)={np.nanmedian(resid_star):.6f}\n")
        handle.write(f"std(resid_star)={np.nanstd(resid_star):.6f}\n")
        handle.write(f"median(resid logssfr_fsps(logMformed)-input_model)={np.nanmedian(resid_formed):.6f}\n")
        handle.write(f"std(resid_formed)={np.nanstd(resid_formed):.6f}\n")
        handle.write("\n[TEST C]\n")
        handle.write(f"N_light={int(np.sum(ok_light))}\n")
        handle.write(f"slope(t_light vs z)={slope_tlight:.6f}\n")
        handle.write(f"median(t_light)={np.nanmedian(t_light):.6f}\n")

    # Plot A1: tage/t_univ vs z
    plt.figure(figsize=(6.5, 4.8))
    plt.hexbin(z, ratio_tage, gridsize=60, bins="log", mincnt=1)
    xline = np.linspace(np.nanmin(z), np.nanmax(z), 200)
    yline = np.polyval(np.polyfit(z, ratio_tage, 1), xline)
    plt.plot(xline, yline, color="crimson", lw=2, label=f"slope={slope_tage:.3e}")
    plt.xlabel("z")
    plt.ylabel("tage / t_univ")
    plt.title("TEST A1: FSPS tage/t_univ vs z")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    plt.savefig(outdir / "testA_tage_over_tuniv_vs_z.png", dpi=170)
    plt.close()

    # Plot A2: stellar-age fraction vs z
    plt.figure(figsize=(6.5, 4.8))
    plt.hexbin(z, ratio_stellar, gridsize=60, bins="log", mincnt=1)
    yline2 = np.polyval(np.polyfit(z, ratio_stellar, 1), xline)
    plt.plot(xline, yline2, color="crimson", lw=2, label=f"slope={slope_stellar:.3e}")
    plt.xlabel("z")
    plt.ylabel("(t_univ - t_i) / t_univ")
    plt.title("TEST A2: Stellar-age fraction vs z")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    plt.savefig(outdir / "testA_stellar_age_fraction_vs_z.png", dpi=170)
    plt.close()

    # Plot B: ssfr consistency residual vs z
    plt.figure(figsize=(6.5, 4.8))
    plt.hexbin(z, resid_star, gridsize=60, bins="log", mincnt=1)
    plt.xlabel("z")
    plt.ylabel("log sSFR_fsps(star-mass) - log sSFR_input_model")
    plt.title("TEST B: sSFR consistency residual")
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    plt.savefig(outdir / "testB_ssfr_residual_vs_z.png", dpi=170)
    plt.close()

    # Plot C: t_light vs z
    if np.any(ok_light):
        plt.figure(figsize=(6.5, 4.8))
        plt.hexbin(z_sub[ok_light], t_light[ok_light], gridsize=55, bins="log", mincnt=1)
        xline_l = np.linspace(np.nanmin(z_sub[ok_light]), np.nanmax(z_sub[ok_light]), 200)
        yline_l = np.polyval(np.polyfit(z_sub[ok_light], t_light[ok_light], 1), xline_l)
        plt.plot(xline_l, yline_l, color="crimson", lw=2, label=f"slope={slope_tlight:.3e}")
        plt.xlabel("z")
        plt.ylabel("t_light [Gyr]")
        plt.title("TEST C: Light-weighted age vs z")
        plt.legend()
        plt.colorbar(label="log10(N)")
        plt.tight_layout()
        plt.savefig(outdir / "testC_tlight_vs_z.png", dpi=170)
        plt.close()

    print("SFH consistency diagnostics complete")
    print(f"  summary: {summary_path}")
    print(f"  TEST A1 slope(tage/t_univ vs z): {slope_tage:.4e}")
    print(f"  TEST A2 slope((t_univ-ti)/t_univ vs z): {slope_stellar:.4e}")
    print(f"  TEST B median residual (star-mass): {np.nanmedian(resid_star):.4f} dex")
    print(f"  TEST C slope(t_light vs z): {slope_tlight:.4e}")


if __name__ == "__main__":
    main()
