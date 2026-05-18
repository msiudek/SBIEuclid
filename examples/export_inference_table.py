
import numpy as np
import pandas as pd
from astropy.table import Table

# Load inference results
inference = np.load('sbi-logs/inference_desi_v5.1/inference_results.npz')
euclid_id = inference['euclid_id'] 
desi_id = inference['desi_id'] 
logM_sbi = inference['logM_sbi']
logSFR_sbi = inference['logSFR_sbi']
logM_desi = inference['logM_desi']
logSFR_desi = inference['logSFR_desi']

# Build DataFrame
out = pd.DataFrame({
    'euclid_id': euclid_id,
    'desi_id': desi_id,
    'logM_sbi': logM_sbi,
    'logSFR_sbi': logSFR_sbi,
    'logM_desi': logM_desi,
    'logSFR_desi': logSFR_desi,
})

# Save as CSV
out.to_csv('sbi-logs/inference_desi_v5.1/inference_summary.csv', index=False)
print('Saved table to sbi-logs/inference_desi_v5.1/inference_summary.csv')
