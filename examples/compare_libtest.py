"""Combine the FSPS and CIGALE sides of the pure-library mass-loss test.

Reads:
  sbi-logs/libtest_fsps.fits           (from fsps_living_fraction.py, server)
  sbi-logs/cigale_libtest/out/models-block-0.fits  (from `pcigale run`, server)
  sbi-logs/cigale_libtest/libtest_grid_snapped.fits (written by cigale_libtest_config.py)

For each of our N grid objects, finds CIGALE's model matching its own
(snapped tau_main, age_main, metallicity) at its own redshift, reads off
CIGALE's living fraction (stellar.m_star / sfh.integrated, both for 1 Msun
formed), and compares to FSPS's living fraction at the (unsnapped) same
point. The ratio isolates the pure SPS-code mass-loss/normalization
difference -- no photometric fitting involved on either side.

Usage:
    python examples/compare_libtest.py
"""
import numpy as np
from astropy.table import Table, join

FSPS_PATH = "sbi-logs/libtest_fsps.fits"
CIGALE_MODELS_PATH = "sbi-logs/cigale_libtest/out/models-block-0.fits"
SNAPPED_GRID_PATH = "sbi-logs/cigale_libtest/libtest_grid_snapped.fits"


def main():
    fsps_t = Table.read(FSPS_PATH)
    snap_t = Table.read(SNAPPED_GRID_PATH)
    cig = Table.read(CIGALE_MODELS_PATH)

    print(f"FSPS rows: {len(fsps_t)}  CIGALE models: {len(cig)}  objects: {len(snap_t)}")
    print(f"CIGALE model columns: {cig.colnames}")

    # CIGALE column names vary slightly by version; try the likely candidates.
    def pick(colnames, *cands):
        for c in cands:
            if c in colnames:
                return c
        raise KeyError(f"none of {cands} found in {colnames}")

    c_id = pick(cig.colnames, "id")
    c_tau = pick(cig.colnames, "best.sfh.tau_main", "sfh.tau_main")
    c_age = pick(cig.colnames, "best.sfh.age_main", "sfh.age_main")
    c_met = pick(cig.colnames, "best.stellar.metallicity", "stellar.metallicity")
    c_mstar = pick(cig.colnames, "best.stellar.m_star", "stellar.m_star")
    c_mform = pick(cig.colnames, "best.sfh.integrated", "sfh.integrated")

    cig_by_id = {}
    for row in cig:
        cig_by_id.setdefault(int(row[c_id]), []).append(row)

    n = len(snap_t)
    ratio = np.full(n, np.nan)
    fsps_lf = np.full(n, np.nan)
    cig_lf = np.full(n, np.nan)
    z = np.asarray(snap_t["z"])

    for i in range(n):
        oid = int(snap_t["id"][i])
        rows = cig_by_id.get(oid, [])
        if not rows:
            continue
        tau_t, age_t, met_t = (snap_t["tau_main_myr_snapped"][i],
                                snap_t["age_main_myr_snapped"][i],
                                snap_t["metallicity_snapped"][i])
        best = min(rows, key=lambda r: abs(r[c_tau] - tau_t) + abs(r[c_age] - age_t)
                   + 1e3 * abs(r[c_met] - met_t))
        cig_lf[i] = float(best[c_mstar]) / max(float(best[c_mform]), 1e-30)
        fsps_lf[i] = fsps_t["fsps_living_frac"][i]
        ratio[i] = fsps_lf[i] / cig_lf[i] if cig_lf[i] > 0 else np.nan

    ok = np.isfinite(ratio)
    print(f"\nmatched: {ok.sum()}/{n}")
    print(f"FSPS living fraction   median = {np.nanmedian(fsps_lf):.4f}")
    print(f"CIGALE living fraction median = {np.nanmedian(cig_lf):.4f}")
    print(f"ratio FSPS/CIGALE      median = {np.nanmedian(ratio):.4f}  "
          f"-> {np.log10(np.nanmedian(ratio)):+.3f} dex")
    print("(positive dex = FSPS assigns MORE surviving mass than CIGALE/BC03 "
          "for the identical formed-mass-normalized SFH)")

    print("\nby z-bin:")
    for zlo, zhi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5)]:
        s = ok & (z >= zlo) & (z < zhi)
        if s.sum() > 10:
            print(f"  z {zlo}-{zhi}: dex = {np.log10(np.median(ratio[s])):+.3f}  n={s.sum()}")

    stellar_age = np.asarray(snap_t["stellar_age_gyr"])
    print("\nby stellar age:")
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 13)]:
        s = ok & (stellar_age >= lo) & (stellar_age < hi)
        if s.sum() > 10:
            print(f"  age {lo}-{hi} Gyr: dex = {np.log10(np.median(ratio[s])):+.3f}  n={s.sum()}")


if __name__ == "__main__":
    main()
