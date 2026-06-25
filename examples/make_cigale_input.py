"""
Build CIGALE inputs for the Euclid-bands-only "bands vs method" test, on the
EXACT galaxies SBI ran on (inference_euclid_v3): only the 10 Euclid-pipeline
bands SBI saw (VIS=psf, others=templfit), fixed redshift = z_lephare.

Compare CIGALE(Euclid bands) to full-band LePhare and to SBI on identical
objects: match LePhare -> bias is SBI-specific; over-estimate like SBI -> bias
is fundamental to Euclid band coverage.

Outputs (sbi-logs/cigale_euclidbands/):
  cigale_euclid_input.fits     CIGALE photometry table, fluxes in mJy (CIGALE default)
  cigale_euclid_reference.csv  id, z, logM_lephare, logM_sbi for the comparison
  filters/<Name>.dat           CIGALE filter files (same SVO curves as the SBI atlas),
                               same format as the user's existing HSC_g.dat (Å + 3-line header)

Band naming matches the user's CIGALE DB convention:
  Euclid_VIS, Euclid_NISP_Y/J/H, HSC_g/z, DECam_g/r/i/z
"""

import glob
from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
OBSF = ROOT / "obs" / "obs_properties"
CAT = OBSF / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"
NPZ_DIR = ROOT / "sbi-logs" / "inference_euclid_v3"
OUT = ROOT / "sbi-logs" / "cigale_euclidbands"

UJY_TO_MJY = 1.0e-3  # CIGALE input is mJy; catalog fluxes are µJy

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
        wl, tr = np.loadtxt(OBSF / src, unpack=True)  # Å, transmission
        with open(fdir / f"{name}.dat", "w") as fh:
            fh.write(f"# {name}\n# photon\n# {facility}\n")
            for w, t in zip(wl, tr):
                fh.write(f"{w:.4f} {t:.6f}\n")
    print(f"wrote {len(BANDS)} CIGALE filter files -> {fdir}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_filter_files()

    d = np.load(glob.glob(str(NPZ_DIR / "*.npz"))[0])
    sel = np.asarray(d["selected_indices"], dtype=int)
    logM_sbi = np.asarray(d["logM_sbi"], dtype=float)
    logM_ref = np.asarray(d["logM_cosmosweb"], dtype=float)

    cat = Table.read(CAT)
    z = np.asarray(cat["z_lephare"], dtype=float)[sel]
    cosweb_id = np.asarray(cat["cosweb_id"])[sel] if "cosweb_id" in cat.colnames else sel

    out = Table()
    out["id"] = sel.astype(np.int64)
    out["redshift"] = np.round(z, 5)
    for name, fcol, ecol, _, _ in BANDS:
        f = np.asarray(cat[fcol], dtype=float)[sel] * UJY_TO_MJY
        e = np.asarray(cat[ecol], dtype=float)[sel] * UJY_TO_MJY
        bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
        out[name] = np.where(bad, np.nan, f)
        out[name + "_err"] = np.where(bad, np.nan, e)
    out.write(OUT / "cigale_euclid_input.fits", overwrite=True)

    ref = Table()
    ref["id"] = sel.astype(np.int64)
    ref["cosweb_id"] = cosweb_id
    ref["redshift"] = z
    ref["logM_lephare"] = logM_ref
    ref["logM_sbi"] = logM_sbi
    ref.write(OUT / "cigale_euclid_reference.csv", overwrite=True)

    print(f"wrote {OUT/'cigale_euclid_input.fits'}  (N={len(sel)}, 10 bands, mJy)")
    print(f"wrote {OUT/'cigale_euclid_reference.csv'}")
    print(f"redshift z_lephare range [{np.nanmin(z):.2f}, {np.nanmax(z):.2f}]")
    print("bands:", ", ".join(n for n, *_ in BANDS))


if __name__ == "__main__":
    main()
