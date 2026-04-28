from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

# =============================
# LOAD DATA
# =============================
euclid = Table.read("../COSMOS_DEEP_PHZ.fits")

desi = Table.read("~/myspace/projects/DESI/02_mass_estimations/01_Catalogs/01_PhysPropCat/Ironv1.2/IronPhysProp_v1.2.fits")

print("Initial DESI:", len(desi))

# =============================
# CLEAN SAMPLE (HIGH PURITY)
# =============================
mask = (
    (desi['CHI2'] <=17) &
    (desi['LOGM'] > 0)
)

desi = desi[mask]

print("Clean DESI:", len(desi))

# =============================
# COORDINATES
# =============================
coords_desi = SkyCoord(desi['RA'], desi['DEC'], unit='deg')
coords_euclid = SkyCoord(euclid['right_ascension'], euclid['declination'], unit='deg')

# =============================
# SYMMETRIC MATCH
# =============================
idx_se, d2d_se, _ = coords_desi.match_to_catalog_sky(coords_euclid)
idx_es, d2d_es, _ = coords_euclid.match_to_catalog_sky(coords_desi)

radius = 1.0 * u.arcsec

sym_mask = (d2d_se < radius) & (idx_es[idx_se] == np.arange(len(desi)))

desi_m = desi[sym_mask]
euclid_m = euclid[idx_se[sym_mask]]
sep = d2d_se[sym_mask]

print("Matched:", len(desi_m))

# =============================
# BUILD OUTPUT
# =============================
out = Table()

# IDs
out['desi_id'] = desi_m['TARGETID']
out['euclid_id'] = euclid_m['object_id']


# =============================
# REDSHIFT & PHYSICAL PROPERTIES
# =============================

# LePhare
out['z_desi'] = desi_m['Z']
out['type_desi'] = desi_m['SPECTYPE']
out['logM_desi_Cigale'] = desi_m['LOGM']
out['logM_err_desi_Cigale'] = desi_m['LOGM_ERR']
out['logSFR_desi_Cigale'] = desi_m['LOGSFR']
out['logSFR_err_desi_Cigale'] = desi_m['LOGSFR_ERR']


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
out.write("matched_euclid_desi.fits", overwrite=True)

print("Final sample:", len(out))
