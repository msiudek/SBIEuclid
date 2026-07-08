"""CIGALE side of the pure-library mass-loss test. Builds pcigale.ini + data
file for `analysis_method = savefluxes` (no fitting, no observed fluxes
needed -- CIGALE just computes and dumps every model in the grid). Run on
the server, in the environment where pcigale is installed (see
sbi-logs/cigale_khostovan_matched/RUN_COMMANDS.md for the proven CLI pattern
used elsewhere in this repo).

Grid = the union of the (tau, stellar_age) pairs from gen_libtest_grid.py,
rounded to keep the CIGALE grid a manageable size, with NO dust and NO
nebular emission (matching fsps_living_fraction.py's add_neb_emission=False,
dust2=0.0) so this isolates the stellar-population mass-loss convention only.
Metallicity is snapped to CIGALE's standard BC03 grid (0.0004, 0.004, 0.008,
0.02, 0.05).

REDSHIFT: an EXPLICIT small grid (quantile bins of our sample), NOT one
distinct value per object. Leaving [[redshifting]] redshift= blank + a
data_file with 400 distinct per-object redshifts blew up to 15.5M models
(400 x 38784) and crashed savefluxes (ObservationsManager needs a params
object built from the SED grid alone; the "redshift from data_file" trick
is a pdf_analysis feature, not reliably supported by savefluxes). With an
explicit z grid, CIGALE computes each (tau,age,met,z) combo ONCE; objects
are matched to their NEAREST grid point afterward in compare_libtest.py.

Usage:
    python examples/cigale_libtest_config.py \
        --grid sbi-logs/libtest_grid.fits \
        --out sbi-logs/cigale_libtest

Then on the server (pcigale environment):
    cd sbi-logs/cigale_libtest
    pcigale run
    # output: out/models-block-0.fits  (one row per computed model)
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

BC03_MET_GRID = np.array([0.0004, 0.004, 0.008, 0.02, 0.05])

PCIGALE_INI_TEMPLATE = """data_file = data.fits
parameters_file =
sed_modules = sfhdelayed, bc03, redshifting
analysis_method = savefluxes
cores = 8

[sed_modules_params]
  [[sfhdelayed]]
    tau_main = {tau_list}
    age_main = {age_list}
    tau_burst = 50.0
    age_burst = 20.0
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True
  [[bc03]]
    imf = 1
    metallicity = {met_list}
    separation_age = 10
  [[redshifting]]
    redshift = {z_list}

[analysis_params]
  bands = Euclid_VIS, Euclid_NISP_Y, Euclid_NISP_J, Euclid_NISP_H, HSC_g, HSC_z, DECam_g, DECam_r, DECam_i, DECam_z
  properties = stellar.m_star, sfh.integrated, stellar.age_m_star
  additionalerror = 0.0
  save_sed = False
  blocks = 1
"""


def nearest_met(m):
    return float(BC03_MET_GRID[np.argmin(np.abs(BC03_MET_GRID - m))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="sbi-logs/libtest_grid.fits")
    ap.add_argument("--out", default="sbi-logs/cigale_libtest")
    ap.add_argument("--round-myr", type=float, default=250.0,
                     help="rounding step for tau_main/age_main [Myr] to bound grid size")
    ap.add_argument("--n-z-bins", type=int, default=15,
                     help="number of explicit redshift grid points (quantiles of the sample)")
    args = ap.parse_args()

    t = Table.read(args.grid)
    tau_myr = np.round(np.asarray(t["tau_main_myr"]) / args.round_myr) * args.round_myr
    age_myr = np.round(np.asarray(t["age_main_myr"]) / args.round_myr) * args.round_myr
    tau_myr = np.clip(tau_myr, 100, 10000)
    age_myr = np.clip(age_myr, args.round_myr, 13000)
    met_abs = np.array([nearest_met(m) for m in t["metallicity_absolute"]])

    z_grid = np.unique(np.round(
        np.quantile(np.asarray(t["z"]), np.linspace(0, 1, args.n_z_bins)), 3))
    z_snapped = z_grid[np.argmin(np.abs(np.asarray(t["z"])[:, None] - z_grid[None, :]), axis=1)]

    t["tau_main_myr_snapped"] = tau_myr
    t["age_main_myr_snapped"] = age_myr
    t["metallicity_snapped"] = met_abs
    t["z_snapped"] = z_snapped

    tau_vals = sorted(set(tau_myr.tolist()))
    age_vals = sorted(set(age_myr.tolist()))
    met_vals = sorted(set(met_abs.tolist()))
    n_models = len(tau_vals) * len(age_vals) * len(met_vals) * len(z_grid)
    print(f"grid axes: {len(tau_vals)} tau x {len(age_vals)} age x {len(met_vals)} met "
          f"x {len(z_grid)} z = {n_models} models total (shared across all {len(t)} objects)")
    if n_models > 200_000:
        print(f"WARNING: {n_models} models is large; increase --round-myr or reduce --n-z-bins.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ini = PCIGALE_INI_TEMPLATE.format(
        tau_list=", ".join(f"{v:.0f}" for v in tau_vals),
        age_list=", ".join(f"{v:.0f}" for v in age_vals),
        met_list=", ".join(f"{v:g}" for v in met_vals),
        z_list=", ".join(f"{v:g}" for v in z_grid.tolist()),
    )
    (outdir / "pcigale.ini").write_text(ini)

    data = Table()
    data["id"] = t["id"]
    data["redshift"] = t["z"]  # not used as a fit axis; z is fixed by the module grid above
    data.write(outdir / "data.fits", overwrite=True)

    t.write(outdir / "libtest_grid_snapped.fits", overwrite=True)

    print(f"\nwrote {outdir/'pcigale.ini'} and {outdir/'data.fits'}")
    print(f"snapped grid (for matching after the run) saved to "
          f"{outdir/'libtest_grid_snapped.fits'}")
    print("\nNEXT (on the server, in the pcigale environment):")
    print(f"  cd {outdir}")
    print("  pcigale run")
    print(f"  -> out/models-block-0.fits has one row per (tau,age,met,z) model "
          f"({n_models} rows total, NOT per object); compare_libtest.py does the "
          f"nearest-point match to each of our {len(t)} objects.")


if __name__ == "__main__":
    main()
