"""FSPS side of the pure-library mass-loss test. Run on the SERVER (needs FSPS).

For each row of the grid from gen_libtest_grid.py, build the identical
delayed-tau SFH (ti=0, matching CIGALE's sfhdelayed shape), normalize to
1 Msun FORMED mass, and record FSPS's surviving stellar mass. The ratio
sp.stellar_mass / sp.formed_mass is the living fraction FSPS predicts for
that (tau, age, metallicity) -- no photometry, no fitting, just the mass
budget FSPS's isochrones + mass-loss/remnant treatment produce.

Usage (server, sbi_env active, SPS_HOME set):
    python examples/fsps_living_fraction.py \
        --grid sbi-logs/libtest_grid.fits \
        --out sbi-logs/libtest_fsps.fits
    # optional: --isochrone-type padova_1994 --no-stellar-remnants
    # to also probe those knobs on this clean, direct test.
"""
import argparse

import numpy as np
from astropy.table import Table
from scipy import integrate


def escalon(t, ti):
    return (t >= ti).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="sbi-logs/libtest_grid.fits")
    ap.add_argument("--out", default="sbi-logs/libtest_fsps.fits")
    ap.add_argument("--isochrone-type", default=None,
                    help="e.g. padova_1994 to re-test the June-4 knob on this clean setup")
    ap.add_argument("--no-stellar-remnants", action="store_true")
    args = ap.parse_args()

    import fsps
    kwargs = dict(compute_vega_mags=False, zcontinuous=1, sfh=3, imf_type=1,
                  dust_type=2, dust2=0.0, add_neb_emission=False)
    if args.isochrone_type is not None:
        kwargs["isochrone_type"] = args.isochrone_type
    if args.no_stellar_remnants:
        kwargs["add_stellar_remnants"] = False
    sp = fsps.StellarPopulation(**kwargs)
    print(f"FSPS StellarPopulation kwargs: {kwargs}")

    t = Table.read(args.grid)
    n = len(t)
    living_frac = np.full(n, np.nan)
    mstar = np.full(n, np.nan)
    mformed = np.full(n, np.nan)

    for i in range(n):
        z = float(t["z"][i])
        stellar_age = float(t["stellar_age_gyr"][i])
        tau = float(np.clip(t["tau_gyr"][i], 0.1, 10.0))
        met = float(t["logzsol"][i])

        tax = np.linspace(0.0, max(stellar_age, 1e-3), 1000)
        # SFR(t) ~ t*exp(-t/tau), ti=0, normalized to 1 Msun total FORMED mass
        denom = integrate.quad(lambda x: x * np.exp(-x / tau), tax.min(), tax.max())[0]
        sfh = (tax * np.exp(-tax / tau)) / denom if denom > 0 else np.zeros_like(tax)
        sfh = np.where(np.isnan(sfh) | (sfh < 1e-33), 1e-33, sfh)
        sfh_myr = sfh / 1e9  # Msun/Gyr -> Msun/yr for FSPS tabular SFH

        sp.params["logzsol"] = met
        sp.params["gas_logz"] = met
        sp.params["zred"] = z
        sp.set_tabular_sfh(tax, sfh_myr)
        try:
            sp.get_spectrum(tage=max(stellar_age, 1e-3), peraa=False)
            mstar[i] = sp.stellar_mass
            mformed[i] = sp.formed_mass
            living_frac[i] = sp.stellar_mass / sp.formed_mass
        except Exception as exc:
            print(f"  row {i} failed: {exc}")

        if i % 50 == 0:
            print(f"  {i}/{n}  z={z:.2f} age={stellar_age:.2f} tau={tau:.2f} "
                  f"met={met:+.2f}  living_frac={living_frac[i]:.4f}")

    t["fsps_stellar_mass"] = mstar
    t["fsps_formed_mass"] = mformed
    t["fsps_living_frac"] = living_frac
    t.write(args.out, overwrite=True)
    ok = np.isfinite(living_frac)
    print(f"\nsaved {args.out}  ({ok.sum()}/{n} rows ok)")
    print(f"FSPS living fraction: median={np.median(living_frac[ok]):.4f}  "
          f"p16/p84={np.percentile(living_frac[ok],[16,84]).round(4)}")


if __name__ == "__main__":
    main()
