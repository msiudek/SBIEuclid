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

    # Main band summary + per-band summaries
    rows = []
    for band_name in ["NISP-H", "NISP-J", "NISP-Y"]:
        band_index = BAND_MAP[band_name]
        flux = sed_flux[:, band_index]

        valid = (
            np.isfinite(logm)
            & np.isfinite(zval)
            & np.isfinite(flux)
            & (flux > 0)
            & (zval >= 0)
        )
        if np.sum(valid) < 10:
            continue

        x = zval[valid]
        log_ml = logm[valid] - np.log10(flux[valid])
        slope, intercept = fit_line(x, log_ml)
        rows.append((band_name, slope, intercept, int(np.sum(valid))))

    out_txt = outdir / "diagnose_ml_vs_z.txt"
    with open(out_txt, "w", encoding="utf-8") as handle:
        handle.write("band\tslope_logML_vs_z\tintercept\tN\n")
        for band_name, slope, intercept, npts in rows:
            handle.write(f"{band_name}\t{slope:.6f}\t{intercept:.6f}\t{npts}\n")

    # Plot for NISP-H (requested quick diagnostic)
    h_flux = sed_flux[:, BAND_MAP["NISP-H"]]
    valid_h = (
        np.isfinite(logm)
        & np.isfinite(zval)
        & np.isfinite(h_flux)
        & (h_flux > 0)
        & (zval >= 0)
    )

    xh = zval[valid_h]
    yh = logm[valid_h] - np.log10(h_flux[valid_h])
    slope_h, intercept_h = fit_line(xh, yh)

    plt.figure(figsize=(7, 5))
    plt.hexbin(xh, yh, gridsize=60, bins="log", mincnt=1)
    xline = np.linspace(np.nanmin(xh), np.nanmax(xh), 200)
    plt.plot(xline, slope_h * xline + intercept_h, color="crimson", lw=2,
             label=f"slope={slope_h:.3f}")
    plt.xlabel("z")
    plt.ylabel("log10(M*/L_H)  [L_H in microJy proxy]")
    plt.title("Mock atlas diagnostic: log(M/L_H) vs z")
    plt.legend()
    plt.colorbar(label="log10(N)")
    plt.tight_layout()
    out_plot = outdir / "diagnose_ml_vs_z_NISP_H.png"
    plt.savefig(out_plot, dpi=170)
    plt.close()

    print("\nM/L vs z diagnostic complete")
    for band_name, slope, intercept, npts in rows:
        print(f"  {band_name}: slope(logM/L vs z) = {slope:.4f}  (N={npts})")
    print(f"  plot: {out_plot}")
    print(f"  table: {out_txt}")


if __name__ == "__main__":
    main()
