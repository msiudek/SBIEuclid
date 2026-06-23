"""
Build a combined Euclid + JWST-NIR inference catalog (validation test).

Joins the Euclid↔COSMOS-Web matched catalog (Euclid 10-band photometry +
LePhare reference) to the COSMOS-Web catalog that holds the JWST NIRCam
photometry, via the `cosweb_idx` column, and appends F277W/F444W fluxes under
templfit-style names so the standard Euclid inference can read all 12 bands
with --filter-list filters_to_use_euclid_jwstnir.dat --phot-type templfit.

CRITICAL: `cosweb_idx` must index the SAME COSMOS-Web catalog used when the
match was built. This script VALIDATES the join (z/mass consistency) and aborts
if it looks wrong — a silent mis-join produces corr≈0 and meaningless results.

Run on the server where the correct COSMOS-Web JWST catalog lives:
    python examples/build_euclid_jwstnir_catalog.py \
        --euclid-cat obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits \
        --cosweb-cat <PATH/TO/cosmosweb_catalog_used_for_matching.fits> \
        --cosweb-hdu 1 \
        --out obs/obs_properties/COSMOS-Web/matched_euclid_jwstnir.fits
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table


def col_aper0(cat, name):
    a = np.array(cat[name], dtype=float)
    return a[:, 0] if a.ndim == 2 else a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--euclid-cat", required=True, help="matched_euclid_cosmosweb.fits")
    p.add_argument("--cosweb-cat", required=True,
                   help="COSMOS-Web catalog that cosweb_idx indexes (has JWST NIRCam fluxes)")
    p.add_argument("--cosweb-hdu", type=int, default=1, help="HDU with JWST photometry")
    p.add_argument("--cosweb-zcol", default="zpdf_med",
                   help="redshift col in cosweb-cat for join validation (in its ref HDU)")
    p.add_argument("--cosweb-refhdu", type=int, default=2,
                   help="HDU in cosweb-cat holding zpdf_med/mass_med for validation")
    p.add_argument("--f277-col", default="flux_aper_f277w")
    p.add_argument("--f444-col", default="flux_aper_f444w")
    p.add_argument("--f277-err", default="flux_err_aper_f277w")
    p.add_argument("--f444-err", default="flux_err_aper_f444w")
    p.add_argument("--out", required=True)
    p.add_argument("--max-dz", type=float, default=0.1,
                   help="abort if median|z_euclid - z_cosweb| on join exceeds this")
    return p.parse_args()


def main():
    args = parse_args()
    e = Table.read(args.euclid_cat)
    if "cosweb_idx" not in e.colnames:
        raise KeyError("euclid-cat has no 'cosweb_idx' column to join on")
    ci = np.array(e["cosweb_idx"], dtype=int)

    phot = Table.read(args.cosweb_cat, hdu=args.cosweb_hdu)
    n_master = len(phot)
    if ci.min() < 0 or ci.max() >= n_master:
        raise ValueError(f"cosweb_idx range [{ci.min()},{ci.max()}] out of bounds "
                         f"for cosweb-cat (N={n_master}) — wrong catalog/HDU?")

    # ── validate the join with an independent quantity (redshift) ───────
    try:
        ref = Table.read(args.cosweb_cat, hdu=args.cosweb_refhdu)
        z_cw = np.array(ref[args.cosweb_zcol], dtype=float)[ci]
        z_eu = np.array(e["zfinal"], dtype=float)
        ok = np.isfinite(z_cw) & np.isfinite(z_eu) & (z_eu > 0)
        dz = np.median(np.abs(z_eu[ok] - z_cw[ok]))
        print(f"[join validation] median|z_euclid - z_cosweb| = {dz:.4f} (N={ok.sum()})")
        if dz > args.max_dz:
            raise SystemExit(
                f"ABORT: join looks wrong (median Δz={dz:.3f} > {args.max_dz}). "
                "cosweb_idx does not index this catalog — point --cosweb-cat at the "
                "COSMOS-Web catalog actually used when matched_euclid_cosmosweb was built.")
        print("[join validation] PASSED")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[join validation] skipped ({exc}); proceeding WITHOUT validation — verify manually!")

    # ── append JWST NIR fluxes under templfit-style names ───────────────
    out = e.copy()
    f277 = col_aper0(phot, args.f277_col)[ci]
    f444 = col_aper0(phot, args.f444_col)[ci]
    e277 = col_aper0(phot, args.f277_err)[ci]
    e444 = col_aper0(phot, args.f444_err)[ci]
    out["flux_f277w_templfit"]    = f277
    out["fluxerr_f277w_templfit"] = e277
    out["flux_f444w_templfit"]    = f444
    out["fluxerr_f444w_templfit"] = e444

    det = np.isfinite(f277) & (f277 > 0) & np.isfinite(f444) & (f444 > 0)
    print(f"JWST NIR fluxes attached: {det.sum()}/{len(out)} have valid F277W&F444W")
    out.write(args.out, overwrite=True)
    print(f"✓ wrote {args.out}  (cols: Euclid + flux_f277w_templfit, flux_f444w_templfit)")


if __name__ == "__main__":
    main()
