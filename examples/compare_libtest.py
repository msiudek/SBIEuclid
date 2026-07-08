"""Combine the FSPS and CIGALE sides of the pure-library mass-loss test.

Reads:
  sbi-logs/libtest_fsps.fits           (from fsps_living_fraction.py, server)
  sbi-logs/cigale_libtest/out/models-block-0.fits  (from `pcigale run`, server)
  sbi-logs/cigale_libtest/libtest_grid_snapped.fits (written by cigale_libtest_config.py)

The CIGALE grid is SHARED across all objects (55,680 (tau,age,met,z) models,
not one set per object -- see cigale_libtest_config.py). For each of our N
objects, this finds the CIGALE model nearest in (tau_main, age_main,
metallicity, z) among the full shared table, reads off CIGALE's living
fraction (stellar.m_star / sfh.integrated, both for 1 Msun formed), and
compares to FSPS's living fraction at the (unsnapped) same point. The ratio
isolates the pure SPS-code mass-loss/normalization difference -- no
photometric fitting involved on either side.

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

    c_tau = pick(cig.colnames, "best.sfh.tau_main", "sfh.tau_main")
    c_age = pick(cig.colnames, "best.sfh.age_main", "sfh.age_main")
    c_met = pick(cig.colnames, "best.stellar.metallicity", "stellar.metallicity")
    c_z = pick(cig.colnames, "best.universe.redshift", "universe.redshift", "redshift")
    c_mstar = pick(cig.colnames, "best.stellar.m_star", "stellar.m_star")
    c_mform = pick(cig.colnames, "best.sfh.integrated", "sfh.integrated")

    m_tau = np.asarray(cig[c_tau], dtype=float)
    m_age = np.asarray(cig[c_age], dtype=float)
    m_met = np.asarray(cig[c_met], dtype=float)
    m_z = np.asarray(cig[c_z], dtype=float)
    m_lf = np.asarray(cig[c_mstar], dtype=float) / np.maximum(np.asarray(cig[c_mform], dtype=float), 1e-30)

    n = len(snap_t)
    tau_t = np.asarray(snap_t["tau_main_myr_snapped"], dtype=float)
    age_t = np.asarray(snap_t["age_main_myr_snapped"], dtype=float)
    met_t = np.asarray(snap_t["metallicity_snapped"], dtype=float)
    z_t = np.asarray(snap_t["z_snapped"], dtype=float)
    z = np.asarray(snap_t["z"])

    # normalize each axis by its own grid spacing so no single axis dominates
    # the nearest-neighbor distance, then brute-force match (grid is shared,
    # not per-object, so this is a single N x M search, M~55k -- fast enough).
    def scale(v):
        u = np.unique(v)
        step = np.median(np.diff(np.sort(u))) if len(u) > 1 else 1.0
        return step if step > 0 else 1.0

    s_tau, s_age, s_met, s_z = scale(m_tau), scale(m_age), scale(m_met), scale(m_z)

    ratio = np.full(n, np.nan)
    fsps_lf = np.full(n, np.nan)
    cig_lf = np.full(n, np.nan)

    for i in range(n):
        d = (np.abs(m_tau - tau_t[i]) / s_tau + np.abs(m_age - age_t[i]) / s_age
             + np.abs(m_met - met_t[i]) / s_met + np.abs(m_z - z_t[i]) / s_z)
        j = int(np.argmin(d))
        cig_lf[i] = m_lf[j]
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
