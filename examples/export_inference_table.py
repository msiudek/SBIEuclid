
import numpy as np
import pandas as pd
from astropy.table import Table

# Load inference results
inference = np.load('sbi-logs/inference_cosmosweb_v5.1/inference_results.npz')
sel = inference['selected_indices']  # zero-based row indices into matched catalog
logM_sbi = inference['logM_sbi']
logSFR_sbi = inference['logSFR_sbi']
logM_cosmosweb = inference['logM_cosmosweb']
logSFR_cosmosweb = inference['logSFR_cosmosweb']

# Load the matched catalog to get true euclid_idx values
cat = Table.read('obs/obs_properties/COSMOS-Web/matched_euclid_cosmosweb.fits')
euclid_idx = np.array(cat['euclid_idx'])[sel]

# Build DataFrame
out = pd.DataFrame({
    'index': sel,
    'logM_sbi': logM_sbi,
    'logSFR_sbi': logSFR_sbi,
    'logM_cosmosweb': logM_cosmosweb,
    'logSFR_cosmosweb': logSFR_cosmosweb,
})

# Save as CSV
out.to_csv('sbi-logs/inference_cosmosweb_v5.1/inference_summary.csv', index=False)
print('Saved table to sbi-logs/inference_cosmosweb_v5.1/inference_summary.csv')
