"""
Build CIGALE input for the SPEC-Z apples-to-apples test on Andrea's full Euclid
sample crossmatched to the Khostovan spec-z compilation.

Same 10 Euclid-pipeline bands SBI/CIGALE see (VIS=psf, others=templfit), in mJy,
redshift FIXED to the Khostovan spec-z. Reference masses (full COSMOS bands,
IRAC-constrained) carried for the comparison:
  - khostovan_lp_mass_med        Khostovan LePhare log M*
  - khostovan_cig_logM           Khostovan CIGALE log M* (= log10 bayes.stellar.m_star)
  - pipeline_logM                Euclid pipeline (Phosphoros, no IRAC) log M*

Selection: pass_cuts (pipeline quality) & specz>0 & all 10 templfit bands finite.

Outputs (sbi-logs/cigale_khostovan_specz/):
  cigale_khostovan_input.fits   CIGALE photometry, fluxes in mJy
  cigale_khostovan_reference.csv  object_id, specz, ref masses
  filters/<Name>.dat            CIGALE filter files (same curves as Euclid-bands test)
"""
from pathlib import Path
import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
OBSF = ROOT / "obs" / "obs_properties"
MATCH = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/matched_andrea_khostovan.fits")
CIGREF = Path("/home/msiudek/myspace/projects/COSMOS/Khostovan/cigale_results_specz_compilation_DR1.1.fits")
OUT = ROOT / "sbi-logs" / "cigale_khostovan_specz"

UJY_TO_MJY = 1.0e-3

# CIGALE band name -> (flux col, err col, source curve, facility comment)
BANDS = [
    ("Euclid_VIS",    "flux_vis_psf",             "fluxerr_vis_psf",             "FILTERS_EUCLID/Euclid_VIS.vis.dat", "Euclid VIS"),
    ("Euclid_NISP_Y", "flux_y_templfit",          "fluxerr_y_templfit",          "FILTERS_EUCLID/Euclid_NISP.Y.dat",  "Euclid NISP Y"),
    ("Euclid_NISP_J", "flux_j_templfit",          "fluxerr_j_templfit",          "FILTERS_EUCLID/Euclid_NISP.J.dat",  "Euclid NISP J"),
    ("Euclid_NISP_H", "flux_h_templfit",          "fluxerr_h_templfit",          "FILTERS_EUCLID/Euclid_NISP.H.dat",  "Euclid NISP H"),
    ("HSC_g",         "flux_g_ext_hsc_templfit",  "fluxerr_g_ext_hsc_templfit",  "FILTERS_HSC/Subaru_HSC.g.dat",      "Subaru HSC g"),
    ("HSC_z",         "flux_z_ext_hsc_templfit",  "fluxerr_z_ext_hsc_templfit",  "FILTERS_HSC/Subaru_HSC.z.dat",      "Subaru HSC z"),
    ("DECam_g",       "flux_g_ext_decam_templfit","fluxerr_g_ext_decam_templfit","FILTERS_DECam/CTIO_DECam.g.dat",    "CTIO DECam g"),
    ("DECam_r",       "flux_r_ext_decam_templfit","fluxerr_r_ext_decam_templfit","FILTERS_DECam/CTIO_DECam.r.dat",    "CTIO DECam r"),
    ("DECam_i",       "flux_i_ext_decam_templfit","fluxerr_i_ext_decam_templfit","FILTERS_DECam/CTIO_DECam.i.dat",    "CTIO DECam i"),
    ("DECam_z",       "flux_z_ext_decam_templfit","fluxerr_z_ext_decam_templfit","FILTERS_DECam/CTIO_DECam.z.dat",    "CTIO DECam z"),
]


def write_filter_files():
    fdir = OUT / "filters"
    fdir.mkdir(parents=True, exist_ok=True)
    for name, _, _, src, facility in BANDS:
        wl, tr = np.loadtxt(OBSF / src, unpack=True)
        with open(fdir / f"{name}.dat", "w") as fh:
            fh.write(f"# {name}\n# photon\n# {facility}\n")
            for w, t in zip(wl, tr):
                fh.write(f"{w:.4f} {t:.6f}\n")
    print(f"wrote {len(BANDS)} CIGALE filter files -> {fdir}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_filter_files()

    t = Table.read(MATCH)
    z = np.asarray(t["specz"], dtype=float)
    pc = np.asarray(t["pass_cuts"], dtype=bool)

    phot_ok = np.ones(len(t), bool)
    for _, fcol, ecol, _, _ in BANDS:
        f = np.asarray(t[fcol], dtype=float)
        e = np.asarray(t[ecol], dtype=float)
        phot_ok &= np.isfinite(f) & np.isfinite(e) & (e > 0)

    sel = pc & (z > 0) & phot_ok
    print(f"selected: {int(sel.sum())} galaxies (pass_cuts & specz>0 & 10 bands valid)")

    # join Khostovan reference CIGALE mass by Id_COS20_Classic
    cig = Table.read(CIGREF)
    cig_logM = {int(i): np.log10(m) if m > 0 else np.nan
                for i, m in zip(np.asarray(cig["ID_COS20_Classic"]),
                                np.asarray(cig["bayes.stellar.m_star"], dtype=float))}
    khost_cig = np.array([cig_logM.get(int(i), np.nan) for i in np.asarray(t["Id_COS20_Classic"])])

    ts = t[sel]
    zs = z[sel]

    out = Table()
    out["id"] = np.asarray(ts["object_id"]).astype(np.int64)
    out["redshift"] = np.round(zs, 5)
    for name, fcol, ecol, _, _ in BANDS:
        out[name] = np.asarray(ts[fcol], dtype=float) * UJY_TO_MJY
        out[name + "_err"] = np.asarray(ts[ecol], dtype=float) * UJY_TO_MJY
    out.write(OUT / "cigale_khostovan_input.fits", overwrite=True)

    ref = Table()
    ref["id"] = np.asarray(ts["object_id"]).astype(np.int64)
    ref["specz"] = zs
    ref["specz_conf"] = np.asarray(ts["specz_conf"])
    ref["pipeline_logM"] = np.asarray(ts["pipeline_logM"], dtype=float)
    ref["pipeline_z"] = np.asarray(ts["pipeline_z"], dtype=float)
    ref["khostovan_lp_mass_med"] = np.asarray(ts["khostovan_lp_mass_med"], dtype=float)
    ref["khostovan_cig_logM"] = khost_cig[sel]
    ref.write(OUT / "cigale_khostovan_reference.csv", overwrite=True)

    print(f"wrote {OUT/'cigale_khostovan_input.fits'}  (N={int(sel.sum())}, 10 bands, mJy)")
    print(f"wrote {OUT/'cigale_khostovan_reference.csv'}")
    print(f"spec-z range [{zs.min():.2f}, {zs.max():.2f}], median {np.median(zs):.2f}")
    nlp = np.isfinite(ref['khostovan_lp_mass_med']).sum()
    ncg = np.isfinite(ref['khostovan_cig_logM']).sum()
    print(f"reference masses available: LePhare {nlp}, CIGALE {ncg}")


if __name__ == "__main__":
    main()
