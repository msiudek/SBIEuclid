from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

# =============================
# LOAD DATA
# =============================
euclid = Table.read("../COSMOS_DEEP_PHZ.fits")

cat_photom = Table.read("~/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits", hdu=1)
cat_lephare = Table.read("~/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits", hdu=2)
cat_cigale = Table.read("~/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits", hdu=4)

# =============================
# STACK (ROW-ALIGNED!)
# =============================
cosweb = hstack([cat_photom, cat_lephare, cat_cigale])

print("Initial COSMOS-Web:", len(cosweb))

# =============================
# CLEAN SAMPLE (HIGH PURITY)
# =============================
mask = (
    (cosweb['type'] == 0) &
    (cosweb['warn_flag'] == 0) &
    (np.abs(cosweb['mag_model_f444w']) < 30) &
    (cosweb['flag_star_hsc'] == 0)
)

cosweb = cosweb[mask]

print("Clean COSMOS-Web:", len(cosweb))

# =============================
# COORDINATES
# =============================
coords_cosweb = SkyCoord(cosweb['ra'], cosweb['dec'], unit='deg')
coords_euclid = SkyCoord(euclid['right_ascension'], euclid['declination'], unit='deg')

# =============================
# SYMMETRIC MATCH
# =============================
idx_se, d2d_se, _ = coords_cosweb.match_to_catalog_sky(coords_euclid)
idx_es, d2d_es, _ = coords_euclid.match_to_catalog_sky(coords_cosweb)

radius = 0.5 * u.arcsec

sym_mask = (d2d_se < radius) & (idx_es[idx_se] == np.arange(len(cosweb)))

cosweb_m = cosweb[sym_mask]
euclid_m = euclid[idx_se[sym_mask]]
sep = d2d_se[sym_mask]

print("Matched:", len(cosweb_m))

# =============================
# BUILD OUTPUT
# =============================
out = Table()

# IDs
out['cosweb_id'] = cosweb_m['id']
out['euclid_id'] = euclid_m['object_id']
#out['cosweb_id_khostovan25'] = cosweb_m['id_specz_khostovan25']


# =============================
# REDSHIFT & PHYSICAL PROPERTIES
# =============================

# LePhare
out['z_lephare'] = cosweb_m['zfinal']
out['type_lephare'] = cosweb_m['type']
out['logM_lephare'] = cosweb_m['mass_med']
out['logM_l68_lephare'] = cosweb_m['mass_l68']
out['logM_u68_lephare'] = cosweb_m['mass_u68']
out['logSFR_lephare'] = cosweb_m['sfr_med']
out['logSFR_l68_lephare'] = cosweb_m['sfr_l68']
out['logSFR_u68_lephare'] = cosweb_m['sfr_u68']

# CIGALE
out['logM_cigale'] = np.log10(cosweb_m['mass'])
out['logSFR_cigale'] = np.log10(cosweb_m['sfh_integrated'])

# =============================
# EUCLID TEMPLATE-FIT PHOTOMETRY
# =============================
templ_cols = [c for c in euclid_m.colnames if 'templfit' in c.lower()]

for col in templ_cols:
    out[col] = euclid_m[col]

# VIS PSF
for col in ['flux_vis_psf', 'fluxerr_vis_psf']:
    if col in euclid_m.colnames:
        out[col] = euclid_m[col]


# PHZ columns
templ_cols = [c for c in euclid_m.colnames if 'PHZ' in c.lower()]

for col in templ_cols:
    out[col] = euclid_m[col]

# additional columns
for col in ['patch_id_list']:
    if col in euclid_m.colnames:
        out[col] = euclid_m[col]
        
# =============================
# MATCH QUALITY
# =============================
out['sep_arcsec'] = sep.to(u.arcsec)

# =============================
# SAVE
# =============================
out.write("matched_euclid_cosmosweb.fits", overwrite=True)

print("Final sample:", len(out))
