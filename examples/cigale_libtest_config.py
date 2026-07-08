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

Each object's row 'redshift' in the data file is used directly (the
[[redshifting]] redshift= line is left BLANK, which is CIGALE's convention
for "use the per-object redshift from the data file" -- not a fit axis).

Usage:
    python examples/cigale_libtest_config.py \
        --grid sbi-logs/libtest_grid.fits \
        --out sbi-logs/cigale_libtest

Then on the server:
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
    redshift =

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
    ap.add_argument("--round-myr", type=float, default=50.0,
                     help="rounding step for tau_main/age_main [Myr] to bound grid size")
    args = ap.parse_args()

    t = Table.read(args.grid)
    tau_myr = np.round(np.asarray(t["tau_main_myr"]) / args.round_myr) * args.round_myr
    age_myr = np.round(np.asarray(t["age_main_myr"]) / args.round_myr) * args.round_myr
    tau_myr = np.clip(tau_myr, 100, 10000)
    age_myr = np.clip(age_myr, args.round_myr, 13000)
    met_abs = np.array([nearest_met(m) for m in t["metallicity_absolute"]])

    t["tau_main_myr_snapped"] = tau_myr
    t["age_main_myr_snapped"] = age_myr
    t["metallicity_snapped"] = met_abs

    tau_vals = sorted(set(tau_myr.tolist()))
    age_vals = sorted(set(age_myr.tolist()))
    met_vals = sorted(set(met_abs.tolist()))
    n_models = len(tau_vals) * len(age_vals) * len(met_vals)
    print(f"grid axes: {len(tau_vals)} tau x {len(age_vals)} age x {len(met_vals)} met "
          f"= {n_models} models per object x {len(t)} objects")
    if n_models > 5000:
        print(f"WARNING: {n_models} models/object is large; increase --round-myr to shrink it.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ini = PCIGALE_INI_TEMPLATE.format(
        tau_list=", ".join(f"{v:.0f}" for v in tau_vals),
        age_list=", ".join(f"{v:.0f}" for v in age_vals),
        met_list=", ".join(f"{v:g}" for v in met_vals),
    )
    (outdir / "pcigale.ini").write_text(ini)

    data = Table()
    data["id"] = t["id"]
    data["redshift"] = t["z"]
    data.write(outdir / "data.fits", overwrite=True)

    t.write(outdir / "libtest_grid_snapped.fits", overwrite=True)

    print(f"\nwrote {outdir/'pcigale.ini'} and {outdir/'data.fits'}")
    print(f"snapped grid (for matching after the run) saved to "
          f"{outdir/'libtest_grid_snapped.fits'}")
    print("\nNEXT (on the server, in the pcigale environment):")
    print(f"  cd {outdir}")
    print("  pcigale init   # only if pcigale.ini needs regenerating structure; "
          "otherwise skip -- the file above is already complete")
    print("  pcigale check  # sanity check the config")
    print("  pcigale run")
    print("  -> out/models-block-0.fits has one row per (object, tau, age, met) "
          "model with columns id, best.sfh.tau_main, best.sfh.age_main, "
          "best.bc03.metallicity/best.stellar.metallicity, "
          "best.stellar.m_star, best.sfh.integrated")


if __name__ == "__main__":
    main()
