'''

Cross-matchin Euclid - COSMOS


A likelihood-ratio positional matching using a following apporach:

1. Candidate search within a large radius:
 - 0.8″ (COSMOS2020)
 - 0.5″ (COSMOS-Web)
to ensure high completeness (don’t miss true matches)

2. Likelihood-based matching

For each candidate within the search radius, we compute a positional likelihood:
	𝐿𝑅 ∝ exp(−𝑟^2/2𝜎^2)
 where r: angular separation, σ: combined positional uncertainty assumed to be:
 - ~0.3″ (COSMOS2020)
 - ~0.15″ (COSMOS-Web)

3. The closest match gets the highest LR. For each Euclid source we select the candidate with the highest LR.

4. We then apply a smaller, conservative radius:
 - 0.5″ (COSMOS2020)
 - 0.3″ (COSMOS-Web)
and require:
 - separation < final radius
 - exactly one candidate in search radius
 - LR above threshold (0.5)

To remove ambiguous matches and reduce contamination

5. We enforce unique pairing:
 - each Euclid source → one COSMOS source
 - each COSMOS source → one Euclid source

This is done by: sorting matches by LR (best first) and keeping only the highest-LR match per object.
Thanks to that we avoid duplicates and ensure clean training sample.


For COSMOS-Web, the matching is astrometry-dominated, so a simple separation cut already provides near-optimal reliability and completeness.
'''

import numpy as np
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================
SEARCH_RADIUS_ARCSEC = 0.5
FINAL_RADIUS_ARCSEC = 0.3
SIGMA_POS_ARCSEC = 0.15
LR_THRESHOLD = 0.5

# =============================
# PHOTOMETRY COLUMNS
# =============================
APERTURES = ["2fwhm", "3fwhm"]

FILTER_STEM_TO_COL = {
    "CFHT_MegaCam.r":    "r_ext_megacam",
    "CFHT_MegaCam.u":    "u_ext_megacam",
    "Euclid_NISP.H":     "h",
    "Euclid_NISP.J":     "j",
    "Euclid_NISP.Y":     "y",
    "Euclid_VIS.vis":    "vis",
    "Subaru_HSC.g":      "g_ext_hsc",
    "Subaru_HSC.z":      "z_ext_hsc",
    "PAN-STARRS_PS1.i":  "i_ext_panstarrs",
    "CTIO_DECam.g":      "g_ext_decam",
    "CTIO_DECam.r":      "r_ext_decam",
    "CTIO_DECam.i":      "i_ext_decam",
    "CTIO_DECam.z":      "z_ext_decam",
}

def templfit_col(stem, err=False):
    """Return the template-fit (or PSF for VIS) column name in COSMOS_DEEP.fits."""
    prefix = "fluxerr" if err else "flux"
    if stem == "vis":
        return f"{prefix}_vis_psf"
    return f"{prefix}_{stem}_templfit"

# =============================
# LOAD DATA
# =============================
euclid = Table.read("../COSMOS_DEEP.fits")

cosweb_1 = Table.read("~/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits", hdu=1)
cosweb_2 = Table.read("~/myspace/projects/COSMOS/COSMOSWeb_mastercatalog_v1.fits", hdu=2)

# merge tables
cosweb = hstack([cosweb_1, cosweb_2])

# =============================
# QUALITY CUT (COSMOS-Web)
# =============================
if 'warn_flag' in cosweb.colnames:
    cosweb = cosweb[cosweb['warn_flag'] == 0]

print("COSMOS-Web size:", len(cosweb))
print("Euclid size:", len(euclid))

# =============================
# COORDINATES
# =============================
coords_cosweb = SkyCoord(cosweb['ra'], cosweb['dec'], unit='deg')
coords_euclid = SkyCoord(euclid['right_ascension'], euclid['declination'], unit='deg')

xyz_cosweb = np.vstack(coords_cosweb.cartesian.xyz.value).T
xyz_euclid = np.vstack(coords_euclid.cartesian.xyz.value).T

tree = cKDTree(xyz_cosweb)

# =============================
# MATCHING
# =============================
max_radius_rad = (SEARCH_RADIUS_ARCSEC / 3600) * np.pi / 180
max_dist = 2 * np.sin(max_radius_rad / 2)

idxs = tree.query_ball_point(xyz_euclid, r=max_dist)

matches = []

for i, candidates in enumerate(idxs):
    if len(candidates) == 0:
        continue

    vec_e = xyz_euclid[i]
    vec_f = xyz_cosweb[candidates]

    cosang = np.dot(vec_f, vec_e)
    cosang = np.clip(cosang, -1, 1)

    dists = np.arccos(cosang) * 206264.806

    lr = np.exp(-0.5 * (dists / SIGMA_POS_ARCSEC)**2)

    best = np.argmax(lr)

    matches.append((
        i,
        candidates[best],
        dists[best],
        lr[best],
        len(candidates)
    ))

matches = np.array(matches, dtype=[
    ('euclid_idx', int),
    ('cosweb_idx', int),
    ('sep_arcsec', float),
    ('lr', float),
    ('n_candidates', int)
])

print("Matches found:", len(matches))

# =============================
# BUILD OUTPUT
# =============================
matched_cosweb = cosweb[matches['cosweb_idx']]
matched_euclid = euclid[matches['euclid_idx']]

output = Table()

# IDs
output['euclid_idx'] = matches['euclid_idx']
output['cosweb_idx'] = matches['cosweb_idx']
output['patch_id_list'] = matched_cosweb['tile'] if 'tile' in matched_cosweb.colnames else -1

# matching info
output['sep_arcsec'] = matches['sep_arcsec']
output['lr'] = matches['lr']
output['n_candidates'] = matches['n_candidates']

# =============================
# ADD PHYSICAL PROPERTIES
# =============================
for col in [
    'zfinal', 'zpdf_l68', 'zpdf_u68',
    'mass_med', 'mass_l68', 'mass_u68',
    'sfr_med', 'sfr_l68', 'sfr_u68'
]:
    if col in matched_cosweb.colnames:
        output[col] = matched_cosweb[col]

# =============================
# RE-SYNC TABLES
# =============================
matched_euclid = euclid[output['euclid_idx']]
matched_cosweb = cosweb[output['cosweb_idx']]

# =============================
# ADD FLUXES
# =============================
# Aperture photometry (2fwhm, 3fwhm)
for filt, col_stem in FILTER_STEM_TO_COL.items():
    for ap in APERTURES:
        fcol = f"flux_{col_stem}_{ap}_aper"
        ecol = f"fluxerr_{col_stem}_{ap}_aper"

        if fcol in matched_euclid.colnames:
            output[fcol] = matched_euclid[fcol]

        if ecol in matched_euclid.colnames:
            output[ecol] = matched_euclid[ecol]

# Template-fit photometry (templfit for NISP/ground; psf for VIS)
for filt, col_stem in FILTER_STEM_TO_COL.items():
    fcol = templfit_col(col_stem, err=False)
    ecol = templfit_col(col_stem, err=True)
    if fcol in matched_euclid.colnames:
        output[fcol] = matched_euclid[fcol]
    if ecol in matched_euclid.colnames:
        output[ecol] = matched_euclid[ecol]
# =============================
# FINAL CUTS
# =============================
good = (
    (output['sep_arcsec'] < FINAL_RADIUS_ARCSEC) &
    (output['n_candidates'] == 1) &
    (output['lr'] > LR_THRESHOLD)
)

output = output[good]

print("After cuts:", len(output))

# =============================
# ONE-TO-ONE
# =============================
order = np.argsort(output['lr'])[::-1]
output = output[order]

# enforce one-to-one using COSMOS-Web object index
order = np.argsort(output['lr'])[::-1]
output = output[order]

_, unique_idx = np.unique(output['cosweb_idx'], return_index=True)
output = output[unique_idx]

print("Final (1-1):", len(output))

# =============================
# SAVE
# =============================
output.write("matched_euclid_cosmosweb.fits", overwrite=True)

# =============================
# PLOTS
# =============================
plt.figure()
plt.hist(output['sep_arcsec'], bins=50)
plt.xlabel("Separation (arcsec)")
plt.ylabel("N")
plt.savefig("cosmosweb_sep_hist.png", dpi=150)
plt.close()

plt.figure()
plt.scatter(output['sep_arcsec'], output['lr'], s=1)
plt.xlabel("Separation (arcsec)")
plt.ylabel("LR")
plt.savefig("cosmosweb_lr_vs_sep.png", dpi=150)
plt.close()

plt.figure()
plt.scatter(output['sep_arcsec'], output['lr'], s=1)
plt.xlabel("Separation (arcsec)")
plt.ylabel("LR")
plt.savefig("cosmosweb_lr_vs_sep.png", dpi=150)
plt.close()
