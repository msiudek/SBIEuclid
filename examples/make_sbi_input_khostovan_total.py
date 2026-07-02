"""
Build a TOTAL-ready SBI inference catalog for the Khostovan spec-z sample, i.e. one
carrying the raw 2fwhm aperture columns + the total-scaling ingredients, so it can be
run with `inference_cosmosweb.py --phot-type total` (which rescales on the fly:
flux_total = flux_{stem}_2fwhm_aper * flux_detection_total / flux_vis_2fwhm_aper).

Same 19,264 galaxies / spec-z as the other Khostovan inputs. Columns written:
  object_id, z_lephare(=specz), logM_lephare(=khostovan_lp_mass_med), logM_l68/u68_lephare
  flux_{stem}_2fwhm_aper, fluxerr_{stem}_2fwhm_aper  (10 SBI stems)
  flux_detection_total, flux_vis_2fwhm_aper

Inputs (present locally and on the server, in Andrea/):
  matched_andrea_khostovan.fits  (object_id, specz, khostovan_lp_mass_med)
  phot_andrea-result.fits        (2fwhm aperture + detection_total, joined by object_id)

Run:
  python examples/make_sbi_input_khostovan_total.py
Then infer (v1.0 total model):
  python examples/inference_cosmosweb.py \
    --catalog sbi-logs/cigale_khostovan_specz/sbi_input_khostovan_total_ready.fits \
    --model-name model_euclid_v1.pkl --phot-type total --noise-prefix north_total \
    --observation-space flux --n-gal 25000 --n-bands-min 1 --snr-min 0 \
    --n-samples 200 --sample-with direct --outdir sbi-logs/inference_khostovan_v1
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "sbi-logs" / "cigale_khostovan_specz"
MATCH = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/matched_andrea_khostovan.fits")
PHOT = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/phot_andrea-result.fits")

STEMS = ["h", "j", "y", "vis", "g_ext_hsc", "z_ext_hsc",
         "g_ext_decam", "r_ext_decam", "i_ext_decam", "z_ext_decam"]


def main():
    # 19,264 sample = ids in the reference CSV
    ref = Table.read(D / "cigale_khostovan_reference.csv")
    ref_ids = set(int(i) for i in np.asarray(ref["id"]))

    m = Table.read(MATCH)
    oid = np.asarray(m["object_id"]).astype(np.int64)
    keep = np.array([int(i) in ref_ids for i in oid])
    ms = m[keep]
    koid = np.asarray(ms["object_id"]).astype(np.int64)

    p = fits.open(PHOT)[1].data
    pid = np.asarray(p["object_id"]).astype(np.int64)
    prow = {int(i): k for k, i in enumerate(pid)}
    idx = np.array([prow.get(int(o), -1) for o in koid])
    assert (idx >= 0).all(), f"{(idx < 0).sum()} galaxies missing from phot_andrea"

    out = Table()
    out["object_id"] = koid
    out["z_lephare"] = np.asarray(ms["specz"], dtype=float)
    out["logM_lephare"] = np.asarray(ms["khostovan_lp_mass_med"], dtype=float)
    out["logM_l68_lephare"] = np.full(len(out), np.nan)
    out["logM_u68_lephare"] = np.full(len(out), np.nan)
    for s in STEMS:
        out[f"flux_{s}_2fwhm_aper"] = np.asarray(p[f"flux_{s}_2fwhm_aper"], float)[idx]
        out[f"fluxerr_{s}_2fwhm_aper"] = np.asarray(p[f"fluxerr_{s}_2fwhm_aper"], float)[idx]
    out["flux_detection_total"] = np.asarray(p["flux_detection_total"], float)[idx]
    out["flux_vis_2fwhm_aper"] = np.asarray(p["flux_vis_2fwhm_aper"], float)[idx]

    dst = D / "sbi_input_khostovan_total_ready.fits"
    out.write(dst, overwrite=True)
    s = out["flux_detection_total"] / out["flux_vis_2fwhm_aper"]
    print(f"wrote {dst}  (N={len(out)})")
    print(f"  median detection_total/vis_2fwhm = {np.nanmedian(s):.3f}")
    print(f"  z range [{np.nanmin(out['z_lephare']):.2f}, {np.nanmax(out['z_lephare']):.2f}], "
          f"logM range [{np.nanmin(out['logM_lephare']):.2f}, {np.nanmax(out['logM_lephare']):.2f}]")


if __name__ == "__main__":
    main()
