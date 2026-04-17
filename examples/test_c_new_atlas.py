"""
Run Test C on a regenerated atlas using intrinsic (logM, z) cuts on both mock and real data.

This script compares mock atlas fluxes against COSMOS-Web matched photometry and reports:
- per-band median flux ratios (mock/real)
- tilt slope of log10(ratio) vs band index

Default behavior mirrors prior analysis choices:
- photometry type: templfit
- real detections: SNR >= 2 and flux > 0
- intrinsic cuts: logM in [9.0, 10.8], z in [0.4, 2.5]
- optional sigma-limit preselection on atlas using background_noise_north_<phot>.npy

Usage examples
--------------
python examples/test_c_new_atlas.py \
  --atlas-file library/atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas \
  --phot-type templfit --snr-min 2

python examples/test_c_new_atlas.py \
  --atlas-file library/atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas \
  --phot-type templfit --snr-min 5 --no-sigma-mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table


FILTER_SHORT = [
    "NISP-H",
    "NISP-J",
    "NISP-Y",
    "VIS",
    "HSC-g",
    "HSC-z",
    "DECam-g",
    "DECam-r",
    "DECam-i",
    "DECam-z",
]

FILTER_COL_STEMS = [
    "h",
    "j",
    "y",
    "vis",
    "g_ext_hsc",
    "z_ext_hsc",
    "g_ext_decam",
    "r_ext_decam",
    "i_ext_decam",
    "z_ext_decam",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Test C on a regenerated atlas")
    parser.add_argument(
        "--atlas-file",
        default="library/atlas_obs_euclid_north_validate_100000_Nparam_2.dbatlas",
        help="Path to atlas .dbatlas file",
    )
    parser.add_argument(
        "--matched-fits",
        default="obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits",
        help="Path to matched COSMOS-Web FITS catalog",
    )
    parser.add_argument(
        "--phot-type",
        default="templfit",
        choices=["templfit", "2fwhm", "3fwhm"],
        help="Photometry type for real-data columns and sigma-limit file",
    )
    parser.add_argument("--snr-min", type=float, default=2.0, help="Detection SNR threshold")
    parser.add_argument("--logm-min", type=float, default=9.0)
    parser.add_argument("--logm-max", type=float, default=10.8)
    parser.add_argument("--z-min", type=float, default=0.4)
    parser.add_argument("--z-max", type=float, default=2.5)
    parser.add_argument(
        "--use-sigma-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply atlas preselection median_flux > median(background_noise_north_<phot>)",
    )
    return parser


def load_mock_atlas(atlas_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(atlas_file, "r") as file_handle:
        group = file_handle["data"]
        # Atlas SED is stored in microJy (not magnitudes)
        mock_flux_ujy = np.asarray(group['"sed"'][:], dtype=float)
        mock_logm = np.asarray(group['"mstar"'][:], dtype=float)
        mock_z = np.asarray(group['"zval"'][:], dtype=float)
    return mock_flux_ujy, mock_logm, mock_z


def apply_sigma_limit_mask(mock_flux_ujy: np.ndarray, mock_logm: np.ndarray, mock_z: np.ndarray, phot_type: str, project_root: Path):
    limits_file = project_root / "obs" / "obs_properties" / f"background_noise_north_{phot_type}.npy"
    limits = np.load(limits_file)
    sigma_lim = float(np.nanmedian(limits[: mock_flux_ujy.shape[1]]))

    median_flux = np.nanmedian(mock_flux_ujy, axis=1)
    flux_mask = np.isfinite(median_flux) & (median_flux > sigma_lim)
    return mock_flux_ujy[flux_mask], mock_logm[flux_mask], mock_z[flux_mask], sigma_lim


def load_real_fluxes(cat: Table, phot_type: str, snr_min: float) -> np.ndarray:
    n_gal = len(cat)
    real_flux = np.full((len(FILTER_COL_STEMS), n_gal), np.nan)

    for filter_index, stem in enumerate(FILTER_COL_STEMS):
        if phot_type == "templfit":
            if stem == "vis":
                flux_col = "flux_vis_psf"
                err_col = "fluxerr_vis_psf"
            else:
                flux_col = f"flux_{stem}_templfit"
                err_col = f"fluxerr_{stem}_templfit"
        else:
            flux_col = f"flux_{stem}_{phot_type}_aper"
            err_col = f"fluxerr_{stem}_{phot_type}_aper"

        if flux_col not in cat.colnames or err_col not in cat.colnames:
            continue

        flux = np.asarray(cat[flux_col], dtype=float)
        err = np.asarray(cat[err_col], dtype=float)

        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        snr = np.where(valid, flux / err, np.nan)
        detected = valid & np.isfinite(snr) & (snr >= snr_min) & (flux > 0)

        real_flux[filter_index] = np.where(detected, flux, np.nan)

    return real_flux


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    atlas_file = (project_root / args.atlas_file).resolve() if not Path(args.atlas_file).is_absolute() else Path(args.atlas_file)
    matched_fits = (project_root / args.matched_fits).resolve() if not Path(args.matched_fits).is_absolute() else Path(args.matched_fits)

    if not atlas_file.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_file}")
    if not matched_fits.exists():
        raise FileNotFoundError(f"Matched FITS file not found: {matched_fits}")

    mock_flux_ujy, mock_logm, mock_z = load_mock_atlas(atlas_file)

    if args.use_sigma_mask:
        mock_flux_ujy, mock_logm, mock_z, sigma_lim = apply_sigma_limit_mask(
            mock_flux_ujy, mock_logm, mock_z, args.phot_type, project_root
        )
        print(f"sigma_lim ({args.phot_type}) = {sigma_lim:.4e} uJy")
    else:
        print("sigma_lim mask disabled")

    cat = Table.read(matched_fits)
    real_z = np.asarray(cat["zfinal"], dtype=float)
    real_logm = np.asarray(cat["mass_med"], dtype=float)

    mock_intrinsic = (
        np.isfinite(mock_logm)
        & np.isfinite(mock_z)
        & (mock_logm >= args.logm_min)
        & (mock_logm <= args.logm_max)
        & (mock_z >= args.z_min)
        & (mock_z <= args.z_max)
    )

    real_valid = np.isfinite(real_z) & np.isfinite(real_logm) & (real_z > 0) & (real_logm > 0)
    real_intrinsic = (
        real_valid
        & (real_logm >= args.logm_min)
        & (real_logm <= args.logm_max)
        & (real_z >= args.z_min)
        & (real_z <= args.z_max)
    )

    real_flux = load_real_fluxes(cat, phot_type=args.phot_type, snr_min=args.snr_min)

    mock_flux_cut = mock_flux_ujy[mock_intrinsic].T
    real_flux_cut = real_flux[:, real_intrinsic]

    print(f"atlas file: {atlas_file}")
    print(f"real file:  {matched_fits}")
    print(f"phot_type={args.phot_type}, snr_min={args.snr_min:.1f}")
    print(
        f"intrinsic cuts: logM in [{args.logm_min:.2f}, {args.logm_max:.2f}], "
        f"z in [{args.z_min:.2f}, {args.z_max:.2f}]"
    )
    print(f"mock cut: {int(np.sum(mock_intrinsic))} / {len(mock_intrinsic)}")
    print(f"real cut: {int(np.sum(real_intrinsic))} / {len(real_intrinsic)}")

    print("\n  band        n_mock  med_mock(uJy)   n_real  med_real(uJy)  mock/real")

    ratios = []
    for filter_index, band_name in enumerate(FILTER_SHORT):
        mock_flux_band = mock_flux_cut[filter_index]
        real_flux_band = real_flux_cut[filter_index]

        mock_ok = mock_flux_band[np.isfinite(mock_flux_band) & (mock_flux_band > 0)]
        real_ok = real_flux_band[np.isfinite(real_flux_band) & (real_flux_band > 0)]

        if len(mock_ok) == 0 or len(real_ok) == 0:
            print(f"{band_name:>10s}: no data")
            continue

        mock_median = float(np.nanmedian(mock_ok))
        real_median = float(np.nanmedian(real_ok))
        ratio = mock_median / real_median
        ratios.append((filter_index, ratio))

        print(
            f"{band_name:>10s}: n={len(mock_ok):6d}  {mock_median:.4e}    "
            f"n={len(real_ok):6d}  {real_median:.4e}   {ratio:.4f}"
        )

    if len(ratios) >= 2:
        x_vals = np.array([row[0] for row in ratios], dtype=float)
        y_vals = np.log10(np.array([row[1] for row in ratios], dtype=float))
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        print(f"\nTilt slope log10(ratio) vs band_index = {slope:.4f}")
        print(f"Intercept = {intercept:.4f}")
    else:
        print("\nNot enough valid bands to fit tilt slope")


if __name__ == "__main__":
    main()
