from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.table import join
import matplotlib.pyplot as plt

# --- Paths ---
specz_file = "~/myspace/projects/COSMOS/Khostovan/specz_compilation_COSMOS_DR1.1_unique.fits"
euclid_file = "~/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties/COSMOS_DEEP.fits"
cigale_file = "~/myspace/projects/COSMOS/Khostovan/cigale_results_specz_compilation_DR1.1.fits"
lephare_file = "~/myspace/projects/COSMOS/Khostovan/lephare_results_specz_compilation_DR1.1.fits"

out_file = "~/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties/COSMOS-Khostovan/matched_euclid_Khostovan.fits"
mass_file = "~/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties/COSMOS-Khostovan/mass.png"
sfr_file = "~/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties/COSMOS-Khostovan/sfr.png"
delta_file = "~/myspace/projects/EUCLID/DR1/SBI/obs/obs_properties/COSMOS-Khostovan/delta.png"

# --- Load data ---
spec = Table.read(specz_file)
euclid = Table.read(euclid_file)
cigale = Table.read(cigale_file)
cigale.rename_column('ID_COS20_Classic', 'Id_COS20_Classic')
lep = Table.read(lephare_file)
lep.rename_column('Id', 'Id_COS20_Classic')

# --- Coordinates ---
spec_coord = SkyCoord(
    ra=spec['ra_corrected'] * u.deg,
    dec=spec['dec_corrected'] * u.deg
)

euclid_coord = SkyCoord(
    ra=euclid['right_ascension'] * u.deg,
    dec=euclid['declination'] * u.deg
)

# =========================
# SYMMETRIC MATCHING
# =========================

# spec -> Euclid
idx_se, d2d_se, _ = spec_coord.match_to_catalog_sky(euclid_coord)

# Euclid -> spec
idx_es, d2d_es, _ = euclid_coord.match_to_catalog_sky(spec_coord)

radius = 0.7 * u.arcsec

sym_mask = []

for i, j in enumerate(idx_se):
    # radius cut
    if d2d_se[i] > radius:
        sym_mask.append(False)
        continue

    # symmetric condition
    if idx_es[j] == i:
        sym_mask.append(True)
    else:
        sym_mask.append(False)

sym_mask = np.array(sym_mask)

# Apply mask
spec_m = spec[sym_mask]
euclid_m = euclid[idx_se[sym_mask]]
sep = d2d_se[sym_mask]

print(f"Matched (symmetric): {len(spec_m)}")

# =========================
# BUILD OUTPUT TABLE
# =========================

matched = Table()

# --- Spec-z columns ---
spec_cols = [
    'specz',   
    'flag',
    'Confidence_level',
    'survey',
    'public_or_private',
    'Id_COS20_Classic',
    'ra_corrected',
    'dec_corrected',
    'photoz',
    'photoz_type'
]

for col in spec_cols:
    if col in spec_m.colnames:
        matched[col] = spec_m[col]

# --- Euclid ID ---
matched['object_id'] = euclid_m['object_id']

# =========================
# TMPLFIT (templfit) PHOTOMETRY
# =========================
templ_cols = [c for c in euclid_m.colnames if 'templfit' in c.lower()]

print(f"Found {len(templ_cols)} templfit columns")

for col in templ_cols:
    matched[col] = euclid_m[col]

# =========================
# VIS PSF
# =========================
if 'flux_vis_psf' in euclid_m.colnames:
    matched['flux_vis_psf'] = euclid_m['flux_vis_psf']

if 'fluxerr_vis_psf' in euclid_m.colnames:
    matched['fluxerr_vis_psf'] = euclid_m['fluxerr_vis_psf']

# =========================
# FLAGS
# =========================
flag_cols = [
    'flag_vis', 'flag_y', 'flag_j', 'flag_h',
    'spurious_flag', 'det_quality_flag',
    'patch_id_list'
]

for col in flag_cols:
    if col in euclid_m.colnames:
        matched[col] = euclid_m[col]



# =========================
# LOAD CIGALE
# =========================
# --- JOIN on Id_COS20_Classic ---
matched = join(
    matched,
    cigale,
    keys='Id_COS20_Classic',
    join_type='left',
    table_names=['', 'cig']
)
print(f"CIGALE joined: {len(matched)}")

# =========================
# CIGALE QUANTITIES
# =========================

def safe_log(x):
    return np.where(x > 0, np.log10(x), np.nan)

# --- log quantities ---
matched['Khostovan_logM_Cigale'] = safe_log(matched['bayes.stellar.m_star'])
matched['Khostovan_logSFR_Cigale'] = safe_log(matched['bayes.sfh.sfr'])
matched['Khostovan_logSFR100Myr_Cigale'] = safe_log(matched['bayes.sfh.sfr100Myrs'])

# --- errors (log space) ---
def log_err(x, xerr):
    return np.where(x > 0, xerr / (x * np.log(10)), np.nan)

matched['Khostovan_logMerr_Cigale'] = log_err(
    matched['bayes.stellar.m_star'],
    matched['bayes.stellar.m_star_err']
)

matched['Khostovan_logSFRerr_Cigale'] = log_err(
    matched['bayes.sfh.sfr'],
    matched['bayes.sfh.sfr_err']
)

matched['Khostovan_logSFR100Myr_err_Cigale'] = log_err(
    matched['bayes.sfh.sfr100Myrs'],
    matched['bayes.sfh.sfr100Myrs_err']
)


#print(matched.colnames)
#print(cigale.colnames)

# --- redshift sanity ---
cig_z_col = [c for c in matched.colnames if 'redshift' in c and c != 'redshift'][0]
matched['CIGALE_z_diff'] = np.abs(matched['redshift'] - matched[cig_z_col])

print("CIGALE z diff (max, median):",
      np.nanmax(matched['CIGALE_z_diff']),
      np.nanmedian(matched['CIGALE_z_diff']))
      
# =========================
# LOAD LEPHARE
# =========================

matched = join(
    matched,
    lep,
    keys='Id_COS20_Classic',
    join_type='left',
    table_names=['', 'lep']
)

print(f"LePhare joined: {len(matched)}")

# =========================
# LEPHARE QUANTITIES
# =========================

# already log
matched['Khostovan_logM_LePhare'] = matched['mass_med']
matched['Khostovan_logSFR_LePhare'] = matched['SFR_med']

# --- errors from min/max ---
matched['Khostovan_logMerr_LePhare'] = 0.5 * (
    matched['mass_med_max68'] - matched['mass_med_min68']
)

matched['Khostovan_logSFRerr_LePhare'] = 0.5 * (
    matched['SFR_med_max68'] - matched['SFR_med_min68']
)

# --- redshift sanity ---
matched['LePhare_z_diff'] = np.abs(matched['redshift'] - matched['zs'])

print("LePhare z diff (max, median):",
      np.nanmax(matched['LePhare_z_diff']),
      np.nanmedian(matched['LePhare_z_diff']))
            
# =========================
# SEPARATION
# =========================
matched['separation_arcsec'] = sep.to(u.arcsec)

# =========================
# SAVE
# =========================
matched.write(out_file, overwrite=True)

print(f"Saved {len(matched)} sources to:")
print(out_file)

# =========================
# DIAGNOSTICS
# =========================

plt.figure()
mask = np.isfinite(matched['Khostovan_logM_Cigale']) & np.isfinite(matched['Khostovan_logM_LePhare'])

plt.scatter(
    matched['Khostovan_logM_LePhare'][mask],
    matched['Khostovan_logM_Cigale'][mask],
    s=5, alpha=0.5
)

plt.plot([7, 12], [7, 12], 'k--')
plt.xlabel("LePhare log(M*)")
plt.ylabel("CIGALE log(M*)")
plt.title("Stellar Mass Comparison")

plt.savefig(mass_file)

plt.figure()
mask = np.isfinite(matched['Khostovan_logSFR_Cigale']) & np.isfinite(matched['Khostovan_logSFR_LePhare'])

plt.scatter(
    matched['Khostovan_logSFR_LePhare'][mask],
    matched['Khostovan_logSFR_Cigale'][mask],
    s=5, alpha=0.5
)

plt.plot([-5, 3], [-5, 3], 'k--')
plt.xlabel("LePhare log(SFR)")
plt.ylabel("CIGALE log(SFR)")
plt.title("SFR Comparison")

plt.savefig(sfr_file)

# delta logM
deltaM = matched['Khostovan_logM_Cigale'] - matched['Khostovan_logM_LePhare']

plt.figure()
plt.hist(deltaM[np.isfinite(deltaM)], bins=50)
plt.xlabel("Δ log(M*) (CIGALE - LePhare)")
plt.ylabel("N")
plt.show()

# delta SFR
deltaSFR = matched['Khostovan_logSFR_Cigale'] - matched['Khostovan_logSFR_LePhare']

plt.figure()
plt.hist(deltaSFR[np.isfinite(deltaSFR)], bins=50)
plt.xlabel("Δ log(SFR)")
plt.ylabel("N")
plt.savefig(delta_file)
