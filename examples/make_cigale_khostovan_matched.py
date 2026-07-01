"""
Prepare BOTH CIGALE runs for the Khostovan apples-to-apples band-coverage test,
using a grid that mimics Khostovan et al. (2025, arXiv:2503.00120) assumptions
(reconstructed from cigale_results_specz_compilation_DR1.1.fits best-fit columns
and confirmed in the paper: BC03 / Chabrier / delayed-tau+burst / varied nebular
/ Calzetti with E_BV_factor=1.0 / Draine+2014 dust emission).

The FULL Khostovan grid is ~66.8M models (cluster-only). Here we use a feasible
matched grid (~12.6k models): same modules & physical ranges, coarser nuisance
sampling (logU, E(B-V), tau/age). The SAME grid is applied to:

  euclid/  : Euclid 10 pipeline bands (VIS,Y,J,H,HSC g/z,DECam g/r/i/z)
  full/    : 34 COSMOS bands incl. IRAC (fluxes from the Khostovan FITS)

on the IDENTICAL 19,264 galaxies at the IDENTICAL spec-z, so that
  band-coverage bias = CIGALE(Euclid) - CIGALE(full)   [same code, same priors]
Khostovan's published masses remain an external cross-check.

Outputs under sbi-logs/cigale_khostovan_matched/{euclid,full}/.
Run each on the server with `pcigale run` (see RUN_COMMANDS.md).
"""
from pathlib import Path
import shutil
import numpy as np
from astropy.table import Table
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sbi-logs" / "cigale_khostovan_specz"
OUT = ROOT / "sbi-logs" / "cigale_khostovan_matched"
MATCH = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/matched_andrea_khostovan.fits")
KHOST = Path("/home/msiudek/myspace/projects/COSMOS/Khostovan/cigale_results_specz_compilation_DR1.1.fits")

# ---- Khostovan-matched feasible grid (shared by both runs) -----------------
SED_MODULES = "sfhdelayed, bc03, nebular, dustatt_modified_starburst, dl2014, redshifting"
SED_PARAMS = """  [[sfhdelayed]]
    tau_main = 100, 300, 1000, 3000, 10000
    age_main = 500, 1000, 2000, 4000, 8000, 13000
    tau_burst = 50.0
    age_burst = 20.0
    f_burst = 0.0, 0.1
    sfr_A = 1.0
    normalise = True
  [[bc03]]
    imf = 1
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.05
    separation_age = 10
  [[nebular]]
    logU = -3.0, -2.0, -1.0
    zgas = 0.004, 0.02
    ne = 100
    f_esc = 0.0
    f_dust = 0.0
    lines_width = 300.0
    emission = True
  [[dustatt_modified_starburst]]
    E_BV_lines = 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0
    E_BV_factor = 1.0
    uv_bump_wavelength = 217.5
    uv_bump_width = 35.0
    uv_bump_amplitude = 0.0
    powerlaw_slope = 0.0
    Ext_law_emission_lines = 1
    Rv = 3.1
    filters = B_B90 & V_B90 & FUV
  [[dl2014]]
    qpah = 2.5
    umin = 1.0
    alpha = 2.0
    gamma = 0.1
  [[redshifting]]
    redshift ="""

ANALYSIS = """[analysis_params]
  variables = stellar.m_star, sfh.sfr, sfh.sfr100Myrs, stellar.age_m_star, stellar.metallicity, attenuation.E_BV_lines
  bands =
  save_best_sed = False
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 2
  blocks = 1"""

EUCLID_BANDS = [
    "Euclid_VIS", "Euclid_NISP_Y", "Euclid_NISP_J", "Euclid_NISP_H",
    "HSC_g", "HSC_z", "DECam_g", "DECam_r", "DECam_i", "DECam_z",
]

# Full-band reference = broad bands + IRAC, using ONLY filters present in the
# user's CIGALE database (see full/filters listing). Each entry maps the DB
# filter name (= input column name, must match CIGALE DB) to the Khostovan
# flux column pulled from cigale_results_specz_compilation_DR1.1.fits (mJy).
# COSMOS intermediate/narrow bands + ACS F814W + Suprime-Y are OMITTED (not in
# the user's DB, and they constrain photo-z/emission lines, not stellar mass,
# which is set by the broadband SED shape + rest-frame-NIR IRAC anchor).
FULL_MAP = [
    ("CFHT_u",   "cfht.megacam.u"),
    ("SUBARU_B", "subaru.suprime.B"),
    ("SUBARU_g", "subaru.suprime.g+"),
    ("SUBARU_V", "subaru.suprime.V"),
    ("SUBARU_r", "subaru.suprime.r+"),
    ("SUBARU_i", "subaru.suprime.i+"),
    ("SUBARU_z", "subaru.suprime.z++"),
    ("Y_vista",  "paranal.vircam.Y"),
    ("J_vista",  "paranal.vircam.J"),
    ("H_vista",  "paranal.vircam.H"),
    ("K_vista",  "paranal.vircam.Ks"),
    ("IRAC1",    "spitzer.irac.I1"),
    ("IRAC2",    "spitzer.irac.I2"),
]
FULL_BANDS = [db for db, _ in FULL_MAP]


def bands_line(bands):
    toks = []
    for b in bands:
        toks += [b, b + "_err"]
    return "bands = " + ", ".join(toks)


def write_ini(run_dir, data_file, bands):
    ini = f"""data_file = {data_file}
parameters_file =
sed_modules = {SED_MODULES}
analysis_method = pdf_analysis
cores = 32
{bands_line(bands)}
properties =
additionalerror = 0.1
[sed_modules_params]
{SED_PARAMS}
{ANALYSIS}
"""
    (run_dir / "pcigale.ini").write_text(ini)
    # minimal spec file so `pcigale run` doesn't complain about missing spec
    spec = """data_file = string()
parameters_file = string()
sed_modules = cigale_string_list()
analysis_method = string()
cores = integer(min=1)
bands = cigale_string_list()
properties = cigale_string_list()
additionalerror = float()
"""
    (run_dir / "pcigale.ini.spec").write_text(spec)


def build_euclid():
    d = OUT / "euclid"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "cigale_khostovan_input.fits", d / "cigale_input.fits")
    # copy custom filter curves (Euclid/HSC/DECam) for `pcigale-filters add`
    fdst = d / "filters"
    if fdst.exists():
        shutil.rmtree(fdst)
    shutil.copytree(SRC / "filters", fdst)
    write_ini(d, "cigale_input.fits", EUCLID_BANDS)
    n = len(Table.read(d / "cigale_input.fits"))
    print(f"[euclid] {n} galaxies, {len(EUCLID_BANDS)} bands -> {d}")


def build_full():
    d = OUT / "full"
    d.mkdir(parents=True, exist_ok=True)

    # 19,264 selection = ids in the reference CSV (object_id)
    ref = Table.read(SRC / "cigale_khostovan_reference.csv")
    ref_ids = set(int(i) for i in np.asarray(ref["id"]))

    m = Table.read(MATCH)
    oid = np.asarray(m["object_id"]).astype(np.int64)
    keep = np.array([int(i) in ref_ids for i in oid])
    ms = m[keep]
    print(f"[full] matched {len(ms)} rows to reference ids")

    # Khostovan flux table indexed by COS20 classic id
    k = fits.open(KHOST)[1].data
    kid = np.asarray(k["ID_COS20_Classic"]).astype(np.int64)
    krow = {}
    for j, i in enumerate(kid):
        krow.setdefault(int(i), j)  # first occurrence

    cos_id = np.asarray(ms["Id_COS20_Classic"]).astype(np.int64)
    rows = np.array([krow.get(int(i), -1) for i in cos_id])
    has = rows >= 0
    ms = ms[has]
    rows = rows[has]
    print(f"[full] {len(ms)} galaxies have Khostovan full-band photometry")

    out = Table()
    out["id"] = np.asarray(ms["object_id"]).astype(np.int64)
    out["redshift"] = np.round(np.asarray(ms["specz"], dtype=float), 5)
    kd = k[rows]
    for db_name, flux_col in FULL_MAP:
        f = np.asarray(kd[flux_col], dtype=float)
        e = np.asarray(kd[flux_col + "_err"], dtype=float)
        f[~np.isfinite(f)] = np.nan
        e[~np.isfinite(e)] = np.nan
        out[db_name] = f            # column name == user's CIGALE DB filter name
        out[db_name + "_err"] = e
    out.write(d / "cigale_input.fits", overwrite=True)
    write_ini(d, "cigale_input.fits", FULL_BANDS)
    print(f"[full] wrote {d/'cigale_input.fits'}  ({len(out)} galaxies, {len(FULL_BANDS)} bands)")


def write_readme():
    txt = f"""# Khostovan-matched CIGALE runs (band-coverage test)

Grid mimics Khostovan et al. (2025) assumptions; feasible size (~12,600 models):
  sfhdelayed(+burst), bc03/Chabrier, nebular(varied logU,zgas),
  dustatt_modified_starburst (E_BV_factor=1.0), dl2014, redshifting.

Same grid, same 19,264 galaxies, same spec-z, run twice:
  euclid/  : 10 Euclid pipeline bands           (needs custom filters added)
  full/    : 13 broad bands + IRAC (u,Subaru BgVriz,VISTA YJHKs,IRAC1/2)

full/ uses ONLY filters already in your CIGALE database (CFHT_u, SUBARU_*,
*_vista, IRAC1/2) -> nothing to download, nothing to add. COSMOS medium/narrow
bands + ACS F814W are omitted (they drive photo-z/emission lines, not M*).

## Run on the server (per directory)

    cd euclid
    pcigale-filters add filters/*.dat     # euclid/ ONLY (Euclid/HSC/DECam curves)
    pcigale genconf                       # expands the grid
    pcigale run                           # writes out/results.fits

    cd ../full
    pcigale genconf && pcigale run        # no filter add needed

## After both finish

    out/results.fits  ->  bayes.stellar.m_star   (log10 for mass)
    band coverage = log M*(euclid) - log M*(full)   joined on `id`
"""
    (OUT / "RUN_COMMANDS.md").write_text(txt)
    print(f"wrote {OUT/'RUN_COMMANDS.md'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_euclid()
    build_full()
    write_readme()


if __name__ == "__main__":
    main()
