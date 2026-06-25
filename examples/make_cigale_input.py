"""
Build a CIGALE input table for the EXACT galaxies SBI already ran on
(inference_euclid_v3), using ONLY the 10 Euclid-pipeline bands and the same
fluxes SBI saw (VIS=psf, others=templfit). Fixed redshift = z_lephare.

Purpose: decisive "bands vs method" test. Compare CIGALE(Euclid bands) to
full-band LePhare reference and to SBI on identical objects:
  - CIGALE-Euclid ≈ LePhare  -> Euclid bands DO constrain M/L; bias is SBI-specific.
  - CIGALE-Euclid over-estimates like SBI -> bias is fundamental to Euclid bands.

Outputs (sbi-logs/cigale_euclidbands/):
  cigale_euclid_input.txt      CIGALE photometry table (fluxes in MICROJANSKY)
  cigale_euclid_reference.csv  id, z, logM_lephare, logM_sbi for the comparison

UNITS: fluxes are in MICROJANSKY (µJy), as requested. NOTE: CIGALE's default
input unit is mJy — either configure pcigale for µJy or multiply by 1e-3.
COLUMN NAMES: must match your pcigale filter database (run `pcigale-filters
list`); rename headers / add Euclid filters as needed.
"""

import glob
from pathlib import Path

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"
NPZ_DIR = ROOT / "sbi-logs" / "inference_euclid_v3"
OUT = ROOT / "sbi-logs" / "cigale_euclidbands"

# stem -> (flux col, err col, CIGALE-style band header). VIS uses psf, rest templfit.
BANDS = [
    ("vis",          "flux_vis_psf",            "fluxerr_vis_psf",            "euclid_vis"),
    ("y",            "flux_y_templfit",         "fluxerr_y_templfit",         "euclid_y"),
    ("j",            "flux_j_templfit",         "fluxerr_j_templfit",         "euclid_j"),
    ("h",            "flux_h_templfit",         "fluxerr_h_templfit",         "euclid_h"),
    ("g_ext_hsc",    "flux_g_ext_hsc_templfit", "fluxerr_g_ext_hsc_templfit", "hsc_g"),
    ("z_ext_hsc",    "flux_z_ext_hsc_templfit", "fluxerr_z_ext_hsc_templfit", "hsc_z"),
    ("g_ext_decam",  "flux_g_ext_decam_templfit","fluxerr_g_ext_decam_templfit","decam_g"),
    ("r_ext_decam",  "flux_r_ext_decam_templfit","fluxerr_r_ext_decam_templfit","decam_r"),
    ("i_ext_decam",  "flux_i_ext_decam_templfit","fluxerr_i_ext_decam_templfit","decam_i"),
    ("z_ext_decam",  "flux_z_ext_decam_templfit","fluxerr_z_ext_decam_templfit","decam_z"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(glob.glob(str(NPZ_DIR / "*.npz"))[0])
    sel = np.asarray(d["selected_indices"], dtype=int)
    logM_sbi = np.asarray(d["logM_sbi"], dtype=float)
    logM_ref = np.asarray(d["logM_cosmosweb"], dtype=float)
    z_npz = np.asarray(d["z"], dtype=float)

    cat = Table.read(CAT)
    z = np.asarray(cat["z_lephare"], dtype=float)[sel]
    cosweb_id = np.asarray(cat["cosweb_id"])[sel] if "cosweb_id" in cat.colnames else sel

    # id = matched-catalog row index (== key into the npz arrays, order-aligned)
    ids = sel

    # photometry table (µJy). Invalid (nan flux / err<=0 / nan err) -> 'nan' so CIGALE ignores.
    cols = {}
    for stem, fcol, ecol, name in BANDS:
        f = np.asarray(cat[fcol], dtype=float)[sel]
        e = np.asarray(cat[ecol], dtype=float)[sel]
        bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
        f = np.where(bad, np.nan, f)
        e = np.where(bad, np.nan, e)
        cols[name] = f
        cols[name + "_err"] = e

    # write CIGALE photometry table
    inp = OUT / "cigale_euclid_input.txt"
    headers = ["id", "redshift"]
    for _, _, _, name in BANDS:
        headers += [name, name + "_err"]
    with open(inp, "w") as fh:
        fh.write("# " + " ".join(headers) + "\n")
        for i in range(len(ids)):
            row = [str(int(ids[i])), f"{z[i]:.5f}"]
            for _, _, _, name in BANDS:
                fv, ev = cols[name][i], cols[name + "_err"][i]
                row += ["nan" if not np.isfinite(fv) else f"{fv:.6e}",
                        "nan" if not np.isfinite(ev) else f"{ev:.6e}"]
            fh.write(" ".join(row) + "\n")

    # write reference/comparison companion
    ref = Table()
    ref["id"] = ids
    ref["cosweb_id"] = cosweb_id
    ref["redshift"] = z
    ref["logM_lephare"] = logM_ref
    ref["logM_sbi"] = logM_sbi
    ref.write(OUT / "cigale_euclid_reference.csv", overwrite=True)

    n_valid = np.sum([np.isfinite(cols[n]).sum() for _, _, _, n in BANDS])
    print(f"wrote {inp}  (N={len(ids)} galaxies, 10 bands, µJy)")
    print(f"wrote {OUT/'cigale_euclid_reference.csv'}")
    print(f"redshift: z_lephare, range [{np.nanmin(z):.2f}, {np.nanmax(z):.2f}]")
    print(f"per-band valid fluxes (avg): {n_valid/len(BANDS):.0f}/{len(ids)}")
    print("band headers:", ", ".join(n for _, _, _, n in BANDS))
    print("\nREMINDER: fluxes are µJy (CIGALE default is mJy -> set unit or ×1e-3);")
    print("rename headers to your pcigale-filters DB names (add Euclid filters if missing).")


if __name__ == "__main__":
    main()
