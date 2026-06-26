"""
Build a consistent comparison sample on Andrea's DR1 photometry catalog:
  Euclid pipeline (no IRAC) vs SBI vs CIGALE vs COSMOS-Web LePhare.

Applies the pipeline quality cuts, intersects with the COSMOS-Web match (for the
LePhare reference + templfit photometry SBI/CIGALE use), attaches the pipeline
stellar mass, and emits:
  matched_andrea_subsample.fits  -> run SBI inference on this (--catalog ...)
  cigale_input.fits              -> CIGALE input (templfit, mJy) for the SAME objects
  reference.csv                  -> euclid_id, z, logM_lephare, logM_pipeline

Quality cuts (pipeline paper): SPURIOUS_FLAG=0, DET_QUALITY_FLAG<4,
MUMAX_MINUS_MAG>-2.6, PHZ_CLASSIFICATION=2, PHYS_PARAM_FLAGS=0, H(2fwhm) SNR>=5.
For now uses TEMPLFIT fluxes (the 2fwhm->total recipe is blocked: the detection
2fwhm aperture, the denominator, is not in the catalog).
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from astropy.table import Table

SBI = Path(__file__).resolve().parents[1]
CAT = SBI / "obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits"
ANDREA = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea")
OUT = SBI / "sbi-logs/andrea_consistent"
N_PER_ZBIN = 600
ZBINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 6)]
UJY_TO_MJY = 1e-3

# CIGALE band name -> (flux col, err col) in the matched catalog (templfit; VIS=psf)
BANDS = [
    ("Euclid_VIS", "flux_vis_psf", "fluxerr_vis_psf"),
    ("Euclid_NISP_Y", "flux_y_templfit", "fluxerr_y_templfit"),
    ("Euclid_NISP_J", "flux_j_templfit", "fluxerr_j_templfit"),
    ("Euclid_NISP_H", "flux_h_templfit", "fluxerr_h_templfit"),
    ("HSC_g", "flux_g_ext_hsc_templfit", "fluxerr_g_ext_hsc_templfit"),
    ("HSC_z", "flux_z_ext_hsc_templfit", "fluxerr_z_ext_hsc_templfit"),
    ("DECam_g", "flux_g_ext_decam_templfit", "fluxerr_g_ext_decam_templfit"),
    ("DECam_r", "flux_r_ext_decam_templfit", "fluxerr_r_ext_decam_templfit"),
    ("DECam_i", "flux_i_ext_decam_templfit", "fluxerr_i_ext_decam_templfit"),
    ("DECam_z", "flux_z_ext_decam_templfit", "fluxerr_z_ext_decam_templfit"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat = Table.read(CAT)
    phot = Table.read(ANDREA / "phot_andrea-result.fits")
    phys = Table.read(ANDREA / "phys_andrea-result.fits")

    # index Andrea by object_id
    pid = {int(o): i for i, o in enumerate(np.array(phot["object_id"]))}
    yid = {int(o): i for i, o in enumerate(np.array(phys["object_id"]))}
    eid = np.array(cat["euclid_id"])
    hp = np.array([pid.get(int(e), -1) for e in eid])
    hy = np.array([yid.get(int(e), -1) for e in eid])
    has = (hp >= 0) & (hy >= 0)

    def pcol(name):
        a = np.full(len(eid), np.nan); a[has] = np.array(phot[name])[hp[has]]; return a
    def ycol(name):
        a = np.full(len(eid), np.nan); a[has] = np.array(phys[name])[hy[has]]; return a

    snrH = pcol("flux_h_2fwhm_aper") / pcol("fluxerr_h_2fwhm_aper")
    cuts = has & (pcol("spurious_flag") == 0) & (pcol("det_quality_flag") < 4) \
        & (pcol("mumax_minus_mag") > -2.6) & (ycol("phz_classification") == 2) \
        & (ycol("phys_param_flags") == 0) & (snrH >= 5)
    pipe_logM = ycol("phz_pp_median_stellarmass")
    pipe_z = ycol("phz_pp_median_redshift")

    z = np.array(cat["z_lephare"], dtype=float)
    logM_lp = np.array(cat["logM_lephare"], dtype=float)
    good = cuts & np.isfinite(z) & (z > 0) & np.isfinite(logM_lp) & np.isfinite(pipe_logM)
    print(f"galaxies passing cuts + valid LePhare/pipeline: {good.sum()}")

    # full filtered catalog (+ pipeline columns) for flexibility
    full = cat[good].copy()
    full["logM_pipeline"] = pipe_logM[good]
    full["z_pipeline"] = pipe_z[good]
    full.write(OUT / "matched_andrea_cut_full.fits", overwrite=True)

    # stratified subsample in z_lephare
    rng = np.random.default_rng(0)
    gidx = np.where(good)[0]
    sel = []
    for lo, hi in ZBINS:
        b = gidx[(z[gidx] >= lo) & (z[gidx] < hi)]
        sel.append(rng.choice(b, size=min(N_PER_ZBIN, len(b)), replace=False))
    sel = np.sort(np.concatenate(sel))
    print(f"subsample: {len(sel)} (stratified, ~{N_PER_ZBIN}/zbin)")

    sub = cat[sel].copy()
    sub["logM_pipeline"] = pipe_logM[sel]
    sub["z_pipeline"] = pipe_z[sel]
    sub.write(OUT / "matched_andrea_subsample.fits", overwrite=True)

    # CIGALE input (templfit, mJy) for the subsample
    ci = Table()
    ci["id"] = np.array(cat["euclid_id"])[sel]
    ci["redshift"] = np.round(z[sel], 5)
    for name, fcol, ecol in BANDS:
        f = np.array(cat[fcol], dtype=float)[sel] * UJY_TO_MJY
        e = np.array(cat[ecol], dtype=float)[sel] * UJY_TO_MJY
        bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
        ci[name] = np.where(bad, np.nan, f)
        ci[name + "_err"] = np.where(bad, np.nan, e)
    ci.write(OUT / "cigale_input.fits", overwrite=True)

    ref = Table()
    ref["id"] = np.array(cat["euclid_id"])[sel]
    ref["redshift"] = z[sel]
    ref["logM_lephare"] = logM_lp[sel]
    ref["logM_pipeline"] = pipe_logM[sel]
    ref.write(OUT / "reference.csv", overwrite=True)

    print(f"wrote {OUT}/matched_andrea_subsample.fits, cigale_input.fits, reference.csv")
    print(f"z subsample range [{z[sel].min():.2f}, {z[sel].max():.2f}]")
    print(f"pipeline bias vs LePhare (subsample): {np.median((pipe_logM[sel]-logM_lp[sel])):+.3f}")


if __name__ == "__main__":
    main()
