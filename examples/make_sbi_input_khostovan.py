"""
Build the SBI-inference input catalog for the Khostovan spec-z sample, on the
EXACT same 19,264 galaxies as the CIGALE input (make_cigale_input_khostovan.py).

inference_cosmosweb.py expects LePhare-style column names; we alias:
  z_lephare        = specz                  (condition SBI on the spec-z)
  logM_lephare     = khostovan_lp_mass_med  (reference mass, full-band + IRAC)
  logM_l68/u68     = NaN (no error bars needed for the bias comparison)
The templfit/psf flux columns are already present with the names SBI reads.

Output: sbi-logs/cigale_khostovan_specz/sbi_input_khostovan.fits
Run inference on `main` (euclid_v3) with:
  python examples/inference_cosmosweb.py \
    --catalog sbi-logs/cigale_khostovan_specz/sbi_input_khostovan.fits \
    --model-name model_euclid_v3.pkl --phot-type templfit \
    --noise-prefix north_templfit --observation-space flux \
    --n-gal 25000 --n-bands-min 1 --snr-min 0 --n-samples 200 \
    --outdir sbi-logs/inference_khostovan_v3
"""
from pathlib import Path
import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
MATCH = Path("/home/msiudek/myspace/projects/EUCLID/DR1/Andrea/matched_andrea_khostovan.fits")
OUT = ROOT / "sbi-logs" / "cigale_khostovan_specz"
REF = OUT / "cigale_khostovan_reference.csv"

# same 10 SBI bands as the CIGALE builder (VIS=psf, others templfit)
BANDS = [
    "flux_vis_psf", "flux_y_templfit", "flux_j_templfit", "flux_h_templfit",
    "flux_g_ext_hsc_templfit", "flux_z_ext_hsc_templfit",
    "flux_g_ext_decam_templfit", "flux_r_ext_decam_templfit",
    "flux_i_ext_decam_templfit", "flux_z_ext_decam_templfit",
]


def main():
    t = Table.read(MATCH)
    ref = Table.read(REF)
    ref_ids = set(int(i) for i in np.asarray(ref["id"]))

    oid = np.asarray(t["object_id"]).astype(np.int64)
    keep = np.array([int(i) in ref_ids for i in oid])
    ts = t[keep]
    print(f"selected {len(ts)} rows (match to CIGALE reference ids)")

    ts["z_lephare"] = np.asarray(ts["specz"], dtype=float)
    ts["logM_lephare"] = np.asarray(ts["khostovan_lp_mass_med"], dtype=float)
    ts["logM_l68_lephare"] = np.full(len(ts), np.nan)
    ts["logM_u68_lephare"] = np.full(len(ts), np.nan)

    # sanity: every SBI band present & finite
    for b in BANDS:
        assert b in ts.colnames, f"missing {b}"
    ts.write(OUT / "sbi_input_khostovan.fits", overwrite=True)
    print(f"wrote {OUT/'sbi_input_khostovan.fits'}  (N={len(ts)})")
    print(f"z(specz) range [{np.nanmin(ts['z_lephare']):.3f}, {np.nanmax(ts['z_lephare']):.3f}]")
    print(f"logM_lephare range [{np.nanmin(ts['logM_lephare']):.2f}, {np.nanmax(ts['logM_lephare']):.2f}]")


if __name__ == "__main__":
    main()
