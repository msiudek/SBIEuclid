import numpy as np
from pathlib import Path

base = Path('obs/obs_properties')

# 5σ depths (AB mag) provided by user for the active 10-filter set
depth5_mag = {
    'Euclid_NISP.H': 24.25,  # NISP IH
    'Euclid_NISP.J': 24.25,  # NISP IJ
    'Euclid_NISP.Y': 24.25,  # NISP IY
    'Euclid_VIS.vis': 25.80,
    'Subaru_HSC.g': 26.80,
    'Subaru_HSC.z': 25.90,
    'CTIO_DECam.g': 26.46,
    'CTIO_DECam.r': 25.73,
    'CTIO_DECam.i': 25.54,
    'CTIO_DECam.z': 24.97,
}

# preserve exact order from filters_to_use.dat
order = [
    'Euclid_NISP.H',
    'Euclid_NISP.J',
    'Euclid_NISP.Y',
    'Euclid_VIS.vis',
    'Subaru_HSC.g',
    'Subaru_HSC.z',
    'CTIO_DECam.g',
    'CTIO_DECam.r',
    'CTIO_DECam.i',
    'CTIO_DECam.z',
]

def mag_to_flux_ujy(m):
    return 3631.0 * 1e6 * 10 ** (-0.4 * m)

arr_5sigma_ujy = np.array([mag_to_flux_ujy(depth5_mag[k]) for k in order], dtype=float)

out = base / 'background_noise_north_2fwhm_5sigma.npy'
np.save(out, arr_5sigma_ujy)

print(f'Saved: {out}')
for name, mag, f5 in zip(order, [depth5_mag[k] for k in order], arr_5sigma_ujy):
    print(f'{name:18s}  m5={mag:5.2f}  f5={f5:.6f} uJy  f1={f5/5:.6f} uJy')
PY
