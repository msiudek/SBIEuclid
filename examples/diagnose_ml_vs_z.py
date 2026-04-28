"""
Quick diagnostic: log(M/L) vs z for atlas mocks.

This checks whether M/L increases too strongly with redshift.
By default it uses NISP-H as luminosity proxy and also reports J/Y.

Usage:
  python examples/diagnose_ml_vs_z.py \
    --atlas-file library/atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas \
    --outdir sbi-logs/atlas_diagnostics_ssfrmz_100k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


BAND_MAP = {
    "NISP-H": 0,
    "NISP-J": 1,
    "NISP-Y": 2,
    "VIS": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose log(M/L) vs redshift")
    parser.add_argument(
        "--atlas-file",
        default="library/atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas",
        help="Path to atlas .dbatlas file",
    )
    parser.add_argument(
        "--outdir",
        default="sbi-logs/atlas_diagnostics_ssfrmz_100k",
        help="Output directory for plots",
    )
    return parser


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    atlas_file = root / args.atlas_file if not Path(args.atlas_file).is_absolute() else Path(args.atlas_file)
    outdir = root / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(atlas_file, "r") as file_handle:
        group = file_handle["data"]
        logm = np.asarray(group['"mstar"'][:], dtype=float)
        zval = np.asarray(group['"zval"'][:], dtype=float)
        sed_flux = np.asarray(group['"sed"'][:], dtype=float)  # microJy

    print(f"atlas: {atlas_file}")
    print(f"N total: {len(logm)}")

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z_nonneg = np.clip(zval, 0.0, None)
    d_l_cm = cosmo.luminosity_distance(z_nonneg).to("cm").value
    area_factor = 4.0 * np.pi * d_l_cm**2

    # Main band summary + per-band summaries
    rows = []
    for band_name in ["NISP-H", "NISP-J", "NISP-Y"]:
        band_index = BAND_MAP[band_name]
        flux = sed_flux[:, band_index]
        # Convert observed flux proxy to luminosity-like quantity: L ∝ 4π d_L^2 F
        flux_cgs = flux * 1e-29  # microJy -> erg/s/cm^2/Hz
        lum = flux_cgs * area_factor

        valid = (
            np.isfinite(logm)
            & np.isfinite(zval)
            & np.isfinite(lum)
            & (lum > 0)
            & (zval >= 0)
        )
        if np.sum(valid) < 10:
            continue

        x = zval[valid]
        log_ml = logm[valid] - np.log10(lum[valid])
        slope, intercept = fit_line(x, log_ml)
        rows.append((band_name, slope, intercept, int(np.sum(valid))))

    out_txt = outdir / "diagnose_ml_vs_z.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("band\tslope_logML_vs_z\tintercept\tN\n")
        for band_name, slope, intercept, npts in rows:
            handle.write(f"{band_name}\t{slope:.6f}\t{intercept:.6f}\t{npts}\n")


    # Plot for NISP-H (requested quick diagnostic)
    h_flux = sed_flux[:, BAND_MAP["NISP-H"]]
    h_flux_cgs = h_flux * 1e-29
    h_lum = h_flux_cgs * area_factor
    valid_h = (
        np.isfinite(logm)
        & np.isfinite(zval)
        & np.isfinite(h_lum)
        & (h_lum > 0)
        & (zval >= 0)
    )

    xh = zval[valid_h]
    yh = logm[valid_h] - np.log10(h_lum[valid_h])
    slope_h, intercept_h = fit_line(xh, yh)

    plt.figure(figsize=(7, 5))
    plt.hexbin(xh, yh, gridsize=60, bins="log", mincnt=1)
    xline = np.linspace(np.nanmin(xh), np.nanmax(xh), 200)
    plt.plot(xline, slope_h * xline + intercept_h, color="crimson", lw=2,
             label=f"slope={slope_h:.3f}")
    plt.xlabel("z")
    plt.ylabel("log10(M*/L_H)  [distance-corrected luminosity proxy]")
    plt.title("Mock atlas diagnostic: log(M/L_H) vs z (distance corrected)")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out_plot = outdir / "diagnose_ml_vs_z_NISP_H.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()

    # --- New: M/L vs colors ---
    # Use (Y-J) and (J-H) as color proxies
    y_flux = sed_flux[:, BAND_MAP["NISP-Y"]]
    j_flux = sed_flux[:, BAND_MAP["NISP-J"]]
    h_flux = sed_flux[:, BAND_MAP["NISP-H"]]
    # Convert to AB mag: m_AB = -2.5*log10(f_ujy) + 23.9
    def abmag(f_ujy):
        with np.errstate(divide='ignore', invalid='ignore'):
            return -2.5 * np.log10(f_ujy) + 23.9

    y_mag = abmag(y_flux)
    j_mag = abmag(j_flux)
    h_mag = abmag(h_flux)
    color_yj = y_mag - j_mag
    color_jh = j_mag - h_mag

    # M/L_H as before
    ml_h = logm - np.log10(h_lum)
    valid_color = (
        np.isfinite(ml_h)
        & np.isfinite(color_yj)
        & np.isfinite(color_jh)
        & (h_lum > 0)
    )

    # Plot M/L_H vs (Y-J)
    plt.figure(figsize=(7, 5))
    plt.hexbin(color_yj[valid_color], ml_h[valid_color], gridsize=60, bins="log", mincnt=1)
    plt.xlabel("Y - J [mag]")
    plt.ylabel("log10(M*/L_H)")
    plt.title("Mock atlas: log(M/L_H) vs (Y-J)")
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out_plot_color_yj = outdir / "diagnose_ml_vs_color_YJ.png"
    plt.savefig(out_plot_color_yj, dpi=170)
    plt.close()

    # Plot M/L_H vs (J-H)
    plt.figure(figsize=(7, 5))
    plt.hexbin(color_jh[valid_color], ml_h[valid_color], gridsize=60, bins="log", mincnt=1)
    plt.xlabel("J - H [mag]")
    plt.ylabel("log10(M*/L_H)")
    plt.title("Mock atlas: log(M/L_H) vs (J-H)")
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out_plot_color_jh = outdir / "diagnose_ml_vs_color_JH.png"
    plt.savefig(out_plot_color_jh, dpi=170)
    plt.close()

    # --- New: M/L vs sSFR proxy ---
    # Use (H - VIS) color as a crude sSFR proxy (bluer = higher sSFR)
    vis_flux = sed_flux[:, BAND_MAP["VIS"]]
    vis_mag = abmag(vis_flux)
    color_hvis = h_mag - vis_mag
    valid_ssfr = (
        np.isfinite(ml_h)
        & np.isfinite(color_hvis)
        & (h_lum > 0)
    )
    plt.figure(figsize=(7, 5))
    plt.hexbin(color_hvis[valid_ssfr], ml_h[valid_ssfr], gridsize=60, bins="log", mincnt=1)
    plt.xlabel("H - VIS [mag] (sSFR proxy)")
    plt.ylabel("log10(M*/L_H)")
    plt.title("Mock atlas: log(M/L_H) vs (H-VIS) [sSFR proxy]")
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out_plot_ssfr = outdir / "diagnose_ml_vs_sSFRproxy_H_VIS.png"
    plt.savefig(out_plot_ssfr, dpi=170)
    plt.close()

    print("\nM/L vs z diagnostic complete")
    for band_name, slope, intercept, npts in rows:
        print(f"  {band_name}: slope(logM/L vs z) = {slope:.4f}  (N={npts})")
    print(f"  plot: {out_plot}")
    print(f"  table: {out_txt}")
    print(f"  color plot (Y-J): {out_plot_color_yj}")
    print(f"  color plot (J-H): {out_plot_color_jh}")
    print(f"  sSFR proxy plot (H-VIS): {out_plot_ssfr}")


if __name__ == "__main__":
    main()
