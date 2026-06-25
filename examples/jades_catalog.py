"""
Shared helpers to read JADES DR2 integrated photometry into the band order of a
sbipix filter list. Used by build_jades_noise_model.py and
inference_jades_integrated.py.

JADES catalog (hlsp_jades..._photometry_v2.0_catalog.fits) layout:
  HDU2 FLAG  : ID, RA, DEC, FLAG_ST (star=1), FLAG_BS (bright-star contaminated)
  HDU6 CIRC_CONV : PSF-matched circular fluxes {band}_CIRC{0..6} (+ _ei errors)
  HDU8 KRON_CONV : PSF-matched Kron total fluxes {band}_KRON (+ _KRON_ei errors)
  HDU9 PHOTOZ : EAZY_z_a (photo-z)
Fluxes are in nJy → divide by 1000 for sbipix's µJy convention.
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits

NJY_TO_UJY = 1.0e-3
_BAND_RE = re.compile(r"(F\d{3}(?:LP|[WMN]))")

# aperture -> (hdu_index, flux_col_template, err_col_template)
# Use '_e' (native error) for the error: present for BOTH NIRCam and ACS bands.
# (NIRCam also has '_ei' from the ERR extension, but ACS bands do not, so '_e'
#  is the only uniform, consistent choice across all 19 filters.)
APERTURES = {
    "kron_conv": (8, "{band}_KRON", "{band}_KRON_e"),
    "kron":      (7, "{band}_KRON", "{band}_KRON_e"),
    # CIRC needs an aperture index appended, handled separately
}
CIRC_HDU = {"circ_conv": 6, "circ": 4}


def bands_from_filter_list(filter_list_path):
    """Ordered band tokens (e.g. 'F277W') parsed from a sbipix filter list."""
    bands = []
    for line in Path(filter_list_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        m = _BAND_RE.search(token)
        if not m:
            raise ValueError(f"cannot parse band from filter line: {line!r}")
        bands.append(m.group(1))
    return bands


def _col(hdu, name):
    a = np.array(hdu.data[name], dtype=float)
    return a


def load_photometry(catalog_fits, bands, aperture="kron_conv", circ_index=2):
    """Return flux (N, nband) and fluxerr (N, nband) in µJy, in `bands` order,
    plus a dict with id/ra/dec/z/flag_st/flag_bs.

    aperture: 'kron_conv' (default, total + PSF-matched), 'kron',
              'circ_conv'/'circ' (uses circ_index aperture 0..6).
    """
    with fits.open(catalog_fits) as f:
        flag = f["FLAG"].data
        ids = np.array(flag["ID"])
        ra = np.array(flag["RA"], dtype=float)
        dec = np.array(flag["DEC"], dtype=float)
        flag_st = np.array(flag["FLAG_ST"]) if "FLAG_ST" in flag.columns.names else np.zeros(len(ids))
        flag_bs = np.array(flag["FLAG_BS"]) if "FLAG_BS" in flag.columns.names else np.zeros(len(ids))
        z = np.array(f["PHOTOZ"].data["EAZY_z_a"], dtype=float)

        if aperture in APERTURES:
            hdu_i, fcol, ecol = APERTURES[aperture]
            phot_hdu = f[hdu_i]
            flux = np.column_stack([_col(phot_hdu, fcol.format(band=b)) for b in bands])
            err = np.column_stack([_col(phot_hdu, ecol.format(band=b)) for b in bands])
        elif aperture in CIRC_HDU:
            phot_hdu = f[CIRC_HDU[aperture]]
            flux = np.column_stack([_col(phot_hdu, f"{b}_CIRC{circ_index}") for b in bands])
            err = np.column_stack([_col(phot_hdu, f"{b}_CIRC{circ_index}_e") for b in bands])
        else:
            raise ValueError(f"unknown aperture {aperture!r}")

    flux *= NJY_TO_UJY
    err *= NJY_TO_UJY
    meta = dict(id=ids, ra=ra, dec=dec, z=z, flag_st=flag_st, flag_bs=flag_bs)
    return flux, err, meta


def galaxy_selection(flux, err, meta, bands, snr_bands=("F277W", "F444W"),
                     snr_min=10.0, z_min=0.2, z_max=7.5):
    """Boolean mask: real galaxies, not bright-star-contaminated, well-detected
    in deep NIR, with a valid redshift in range."""
    # FLAG_ST is a bitmask: star=1, galaxy=32768 (NOT 0/1 as the readme says).
    # "not a star" == FLAG_ST != 1. FLAG_BS==0 drops bright-star contamination.
    is_gal = (meta["flag_st"] != 1) & (meta["flag_bs"] == 0)
    zok = np.isfinite(meta["z"]) & (meta["z"] > z_min) & (meta["z"] < z_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err > 0, flux / err, np.nan)
    idx = [bands.index(b) for b in snr_bands if b in bands]
    snr_nir = np.nanmean(snr[:, idx], axis=1)
    detok = np.isfinite(snr_nir) & (snr_nir >= snr_min)
    return is_gal & zok & detok
