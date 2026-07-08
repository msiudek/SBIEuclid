"""Build a matched SFH parameter grid for the FSPS-vs-BC03 pure-library test.

The Khostovan cross-code check (crosscode_fit.py) found +0.240 dex when fitting
CIGALE's own best-fit models (delayed+burst SFH, CIGALE's own grid) with our
FSPS atlas. That comparison bundles two things together: (1) the intrinsic
FSPS-vs-BC03 mass-loss/normalization convention, and (2) any mismatch between
CIGALE's best-fit SFH shape and ours. This script isolates JUST (1): it draws
a grid of (z, tau, stellar_age, metallicity) directly from OUR simulator's own
prior functions (no FSPS/CIGALE needed to draw them), with the onset offset
ti FIXED TO 0 so the SFH functional form is IDENTICAL to CIGALE's sfhdelayed
module (SFR(t) ~ t*exp(-t/tau) for t in [0, age_main]). Dust is set to zero
(E_BV=0) to avoid conflating attenuation-law differences with the pure
stellar-population mass-loss comparison.

For each grid point, both codes are asked: "normalize this exact SFH shape to
1 Msun of FORMED (integrated) mass; what SURVIVING stellar mass do you
predict?" The ratio of surviving fractions (FSPS living_frac / CIGALE
living_frac) at fixed (tau, age, met) is the pure library systematic -- no
photometric fitting, no chi2, no degeneracy.

Companion scripts (run where each code is installed):
    examples/fsps_living_fraction.py   (server sbi_env, needs FSPS)
    examples/cigale_libtest_config.py  (writes pcigale.ini + data_file for
                                         analysis_method=savefluxes; run
                                         `pcigale run` in the CIGALE env)

Usage:
    python examples/gen_libtest_grid.py --n 400 --out sbi-logs/libtest_grid.fits
"""
import argparse

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18

from sbipix.train.simulator import _sample_stellar_age

# Mirrors examples/validate_noise_model.py SIMULATION_CONFIG defaults.
Z_MIN, Z_MAX = 0.02, 4.5          # redshift range actually populated (avoid z~0 edge)
MET_MIN, MET_MAX = -0.8, 0.3      # log Z/Zsun, our current prior (SIMULATION_CONFIG Z_min/max)
TAU_LOGSPAN = (-0.5, 0.3)         # tau = stellar_age * 10**U(TAU_LOGSPAN), matches simulator.py loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of grid points")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sbi-logs/libtest_grid.fits")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # _sample_stellar_age uses np.random internally

    z = rng.uniform(Z_MIN, Z_MAX, args.n)
    age_universe_gyr = Planck18.age(z).value

    stellar_age = np.array([_sample_stellar_age(au) for au in age_universe_gyr])  # Gyr
    tau = stellar_age * 10 ** rng.uniform(*TAU_LOGSPAN, args.n)                    # Gyr
    tau = np.clip(tau, 0.1, 10.0)  # matches sfh_delayed_exponential's internal clip
    met = rng.uniform(MET_MIN, MET_MAX, args.n)   # log Z/Zsun

    t = Table()
    t["id"] = np.arange(args.n)
    t["z"] = z
    t["stellar_age_gyr"] = stellar_age      # = CIGALE age_main/1000 (ti fixed to 0)
    t["tau_gyr"] = tau                       # = CIGALE tau_main/1000
    t["age_main_myr"] = stellar_age * 1e3
    t["tau_main_myr"] = tau * 1e3
    t["logzsol"] = met                       # FSPS logzsol
    t["metallicity_absolute"] = 0.02 * 10 ** met   # CIGALE bc03 'metallicity' (Zsun=0.02, BC03 convention)
    t.write(args.out, overwrite=True)
    print(f"saved {args.out}  (N={args.n})")
    print(f"  z range        [{z.min():.2f}, {z.max():.2f}]")
    print(f"  stellar_age Gyr [{stellar_age.min():.2f}, {stellar_age.max():.2f}]")
    print(f"  tau Gyr         [{tau.min():.2f}, {tau.max():.2f}]")
    print(f"  logZ/Zsun       [{met.min():.2f}, {met.max():.2f}]")
    print("\nNOTE: ti (SFH onset offset) is fixed to 0 for BOTH codes in this "
          "test, matching CIGALE sfhdelayed's functional form exactly. "
          "Dust is off (E_BV=0). This isolates the pure SPS-library mass-loss "
          "convention, not the full analysis-pipeline 'code' term.")


if __name__ == "__main__":
    main()
