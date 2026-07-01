"""
Rebuild the Khostovan SBI + CIGALE(Euclid) inputs with MER-cookbook TOTAL fluxes,
replacing the previous mix (VIS=PSF, others=templfit) that left VIS ~0.57 dex too
faint. Total flux per band (assuming VIS_DET=1, true for this bright spec-z sample):

    s = flux_detection_total / flux_vis_2fwhm_aper          # aperture->total scaling
    flux_XXX_total    = s * flux_XXX_2fwhm_aper
    fluxerr_XXX_total = s * fluxerr_XXX_2fwhm_aper

Ingredients from Andrea/phot_andrea-result.fits (per-band *_2fwhm_aper,
flux_detection_total, flux_vis_2fwhm_aper), joined by object_id.

Outputs (same 19,264 galaxies / spec-z as before):
  sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total.fits
      total fluxes written into the column names inference reads for --phot-type
      templfit: VIS -> flux_vis_psf, others -> flux_{stem}_templfit (+ *err*).
  sbi-logs/cigale_khostovan_matched/euclid_total/{cigale_input.fits,pcigale.ini,...}
      CIGALE Euclid input in mJy with total fluxes; same matched grid.
"""
from pathlib import Path
import shutil
import numpy as np
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "sbi-logs" / "cigale_khostovan_specz"
MEUC = ROOT / "sbi-logs" / "cigale_khostovan_matched" / "euclid"
OUT_CIG = ROOT / "sbi-logs" / "cigale_khostovan_matched" / "euclid_total"
PHOT = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/phot_andrea-result.fits")

# SBI band stem -> 2fwhm flux column in phot_andrea; CIGALE name for the euclid input
BANDS = [
    ("vis",         "flux_vis_2fwhm_aper",          "Euclid_VIS"),
    ("y",           "flux_y_2fwhm_aper",            "Euclid_NISP_Y"),
    ("j",           "flux_j_2fwhm_aper",            "Euclid_NISP_J"),
    ("h",           "flux_h_2fwhm_aper",            "Euclid_NISP_H"),
    ("g_ext_hsc",   "flux_g_ext_hsc_2fwhm_aper",    "HSC_g"),
    ("z_ext_hsc",   "flux_z_ext_hsc_2fwhm_aper",    "HSC_z"),
    ("g_ext_decam", "flux_g_ext_decam_2fwhm_aper",  "DECam_g"),
    ("r_ext_decam", "flux_r_ext_decam_2fwhm_aper",  "DECam_r"),
    ("i_ext_decam", "flux_i_ext_decam_2fwhm_aper",  "DECam_i"),
    ("z_ext_decam", "flux_z_ext_decam_2fwhm_aper",  "DECam_z"),
]
UJY_TO_MJY = 1.0e-3


def sbi_col(stem, err=False):
    prefix = "fluxerr" if err else "flux"
    return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"


def main():
    base = Table.read(D / "sbi_input_khostovan.fits")   # 19,264, has z_lephare/logM_lephare
    oid = np.asarray(base["object_id"]).astype(np.int64)

    p = fits.open(PHOT)[1].data
    pid = np.asarray(p["object_id"]).astype(np.int64)
    prow = {int(i): k for k, i in enumerate(pid)}
    idx = np.array([prow.get(int(o), -1) for o in oid])
    assert (idx >= 0).all(), f"{(idx<0).sum()} galaxies missing from phot_andrea"

    det = np.asarray(p["flux_detection_total"], float)[idx]
    vis2 = np.asarray(p["flux_vis_2fwhm_aper"], float)[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = det / vis2                                   # aperture -> total scaling

    # ---- SBI catalog: overwrite flux columns with TOTAL (uJy), keep aliases ----
    ts = base.copy()
    for stem, a2, _ in BANDS:
        f = np.asarray(p[a2], float)[idx]
        e = np.asarray(p["fluxerr" + a2[4:]], float)[idx]
        ts[sbi_col(stem)] = s * f
        ts[sbi_col(stem, err=True)] = s * e
    ts.write(D / "sbi_input_khostovan_total.fits", overwrite=True)
    # report VIS change
    old_vis = np.asarray(base["flux_vis_psf"], float)
    new_vis = np.asarray(ts["flux_vis_psf"], float)
    m = np.isfinite(old_vis) & (old_vis > 0) & np.isfinite(new_vis) & (new_vis > 0)
    print(f"wrote {D/'sbi_input_khostovan_total.fits'} (N={len(ts)})")
    print(f"  VIS total/old median dex: {np.median(np.log10(new_vis[m]/old_vis[m])):+.3f}")

    # ---- CIGALE Euclid input (mJy) with total fluxes, matched grid ----
    OUT_CIG.mkdir(parents=True, exist_ok=True)
    ci = Table()
    ci["id"] = oid
    ci["redshift"] = np.round(np.asarray(base["z_lephare"], float), 5)
    for stem, a2, cig in BANDS:
        f = np.asarray(p[a2], float)[idx]
        e = np.asarray(p["fluxerr" + a2[4:]], float)[idx]
        ci[cig] = s * f * UJY_TO_MJY
        ci[cig + "_err"] = s * e * UJY_TO_MJY
    ci.write(OUT_CIG / "cigale_input.fits", overwrite=True)
    # reuse the matched-grid pcigale.ini + custom filters from euclid/
    shutil.copy(MEUC / "pcigale.ini", OUT_CIG / "pcigale.ini")
    shutil.copy(MEUC / "pcigale.ini.spec", OUT_CIG / "pcigale.ini.spec")
    if (OUT_CIG / "filters").exists():
        shutil.rmtree(OUT_CIG / "filters")
    shutil.copytree(MEUC / "filters", OUT_CIG / "filters")
    print(f"wrote {OUT_CIG/'cigale_input.fits'} (N={len(ci)}) + pcigale.ini + filters")
    print("CIGALE server run:  cd", OUT_CIG,
          "&& pcigale-filters add filters/*.dat && pcigale genconf && pcigale run")


if __name__ == "__main__":
    main()
