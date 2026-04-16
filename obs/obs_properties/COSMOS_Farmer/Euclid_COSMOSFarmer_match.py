import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================
SEARCH_RADIUS_ARCSEC = 0.8
SIGMA_POS_ARCSEC = 0.3
LR_THRESHOLD = 0.5

# =============================
# Farmer quality mask
# =============================
def farmer_quality_mask(cat):
    mask = np.ones(len(cat), dtype=bool)

    if 'flag_combined' in cat.colnames:
        mask &= (cat['flag_combined'] == 0)

    if 'lp_mass_med' in cat.colnames:
        mask &= np.isfinite(cat['lp_mass_med'])

    if 'lp_zBEST' in cat.colnames:
        mask &= (cat['lp_zBEST'] > 0)

    if 'lp_type' in cat.colnames:
        mask &= (cat['lp_type'] == 0)

    if 'lp_chi2' in cat.colnames:
        mask &= (cat['lp_chi2'] < 10)

    return mask


# =============================
# False match rate
# =============================
def false_match_rate(coords_euclid, coords_farmer, tree, max_dist, n_iter=5):
    rates = []

    for _ in range(n_iter):
        ra_shift = np.random.uniform(-0.1, 0.1, len(coords_euclid)) * u.deg
        dec_shift = np.random.uniform(-0.1, 0.1, len(coords_euclid)) * u.deg

        shifted = SkyCoord(
            ra=coords_euclid.ra + ra_shift,
            dec=coords_euclid.dec + dec_shift
        )

        xyz_shifted = np.vstack(shifted.cartesian.xyz.value).T
        idxs = tree.query_ball_point(xyz_shifted, r=max_dist)

        n_match = sum(len(c) > 0 for c in idxs)
        rates.append(n_match / len(coords_euclid))

    return np.mean(rates)


# =============================
# Reliability / Completeness
# =============================
def compute_reliability_completeness(output, coords_euclid, coords_farmer, tree, max_dist):

    thresholds = np.linspace(0.01, 1.0, 50)
    completeness = []
    reliability = []

    N_total = len(output)
    fmr = false_match_rate(coords_euclid, coords_farmer, tree, max_dist)

    for thr in thresholds:
        sel = output['lr'] > thr
        N_sel = np.sum(sel)

        if N_sel == 0:
            completeness.append(0)
            reliability.append(0)
            continue

        completeness.append(N_sel / N_total)

        N_false = fmr * N_sel
        rel = (N_sel - N_false) / N_sel
        reliability.append(rel)

    return thresholds, np.array(completeness), np.array(reliability)


# =============================
# Load catalogs
# =============================
farmer = Table.read("~/myspace/projects/COSMOS/COSMOS2020_farmer.fits")
euclid = Table.read("COSMOS_DEEP.fits")

#print("Available Farmer columns:")
#print(farmer.colnames)

print("Initial Farmer size:", len(farmer))
print("Initial Euclid size:", len(euclid))

# Apply Farmer cuts
farmer = farmer[farmer_quality_mask(farmer)]
print("Farmer after cuts:", len(farmer))

# =============================
# Coordinates
# =============================
coords_farmer = SkyCoord(farmer['alpha_j2000'], farmer['delta_j2000'], unit='deg')
coords_euclid = SkyCoord(euclid['right_ascension'], euclid['declination'], unit='deg')

xyz_farmer = np.vstack(coords_farmer.cartesian.xyz.value).T
xyz_euclid = np.vstack(coords_euclid.cartesian.xyz.value).T

tree = cKDTree(xyz_farmer)

# =============================
# Matching
# =============================
max_radius_rad = (SEARCH_RADIUS_ARCSEC / 3600) * np.pi / 180
max_dist = 2 * np.sin(max_radius_rad / 2)

idxs = tree.query_ball_point(xyz_euclid, r=max_dist)

matches = []

for i, candidates in enumerate(idxs):
    if len(candidates) == 0:
        continue

    vec_e = xyz_euclid[i]
    vec_f = xyz_farmer[candidates]

    cosang = np.dot(vec_f, vec_e)
    cosang = np.clip(cosang, -1.0, 1.0)

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

print("Total matches (0.8 arcsec):", len(matches))

matches = np.array(matches, dtype=[
    ('euclid_idx', int),
    ('farmer_idx', int),
    ('sep_arcsec', float),
    ('lr', float),
    ('n_candidates', int)
])

# =============================
# Build table
# =============================
matched_farmer = farmer[matches['farmer_idx']]

output = Table()
output['euclid_idx'] = matches['euclid_idx']
output['farmer_idx'] = matches['farmer_idx']
output['sep_arcsec'] = matches['sep_arcsec']
output['lr'] = matches['lr']
output['n_candidates'] = matches['n_candidates']


cols_to_add = [
    # Photo-z (LePhare)
    'lp_zbest',
    'lp_zpdf_l68',
    'lp_zpdf_u68',

    # Stellar mass
    'lp_mass_med',
    'lp_mass_med_min68',
    'lp_mass_med_max68',

    # SFR
    'lp_sfr_med',
    'lp_sfr_med_min68',
    'lp_sfr_med_max68',

    # Optional: EAZY (use later if needed)
    'ez_z_phot',
    'ez_mass',
    'ez_sfr'
]

for col in cols_to_add:
    if col in matched_farmer.colnames:
        output[col] = matched_farmer[col]

# =============================
# Diagnostics plots
# =============================
plt.figure()
plt.hist(output['sep_arcsec'], bins=50)
plt.xlabel("Separation (arcsec)")
plt.ylabel("N")
plt.savefig("sep_hist.png", dpi=150)
plt.close()

plt.figure()
plt.scatter(output['sep_arcsec'], output['lr'], s=1)
plt.xlabel("Separation (arcsec)")
plt.ylabel("LR")
plt.savefig("lr_vs_sep.png", dpi=150)
plt.close()

# =============================
# False match rate
# =============================
fmr = false_match_rate(coords_euclid, coords_farmer, tree, max_dist)
print("False match rate ~", fmr)

# =============================
# Apply cuts
# =============================
good = (
    (output['sep_arcsec'] < 0.5) &
    (output['n_candidates'] == 1) &
    (output['lr'] > LR_THRESHOLD)
)

output = output[good]
print("After quality cuts:", len(output))

# =============================
# 🔥 ONE-TO-ONE MATCHING (NEW)
# =============================

# sort by LR (best matches first)
order = np.argsort(output['lr'])[::-1]
output = output[order]

# keep only one Euclid per Farmer
_, unique_idx = np.unique(output['farmer_idx'], return_index=True)
output = output[unique_idx]

print("After enforcing one-to-one:", len(output))

# =============================
# Check duplicates
# =============================
unique, counts = np.unique(output['farmer_idx'], return_counts=True)
print("Remaining duplicates (should be 0):", np.sum(counts > 1))

# =============================
# Reliability / Completeness
# =============================
thr, comp, rel = compute_reliability_completeness(
    output, coords_euclid, coords_farmer, tree, max_dist
)

plt.figure()
plt.plot(thr, comp, label="Completeness")
plt.plot(thr, rel, label="Reliability")
plt.xlabel("LR threshold")
plt.ylabel("Fraction")
plt.legend()
plt.savefig("reliability_completeness.png", dpi=150)
plt.close()

# =============================
# Save final catalog
# =============================
output.write("matched_euclid_farmer.fits", overwrite=True)
