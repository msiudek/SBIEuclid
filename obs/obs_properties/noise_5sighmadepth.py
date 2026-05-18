"""Generate background_noise_north_{phot_type}.npy from 5σ AB magnitude depths.

The 1σ flux limit stored in background_noise files is f5σ / 5, where f5σ
is the flux corresponding to the 5σ point-source detection magnitude.
These depths apply to all photometry types (2fwhm, 3fwhm, templfit) since
they reflect the survey depth, not aperture-dependent photon noise.

Run from the project root:
    python obs/obs_properties/noise_5sighmadepth.py
"""
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 5σ AB magnitude depths for the active 10-filter set (Euclid+ancillary)
# ---------------------------------------------------------------------------
depth5_mag = {
    'Euclid_NISP.H':  24.25,  # NISP IH
    'Euclid_NISP.J':  24.25,  # NISP IJ
    'Euclid_NISP.Y':  24.25,  # NISP IY
    'Euclid_VIS.vis': 25.80,
    'Subaru_HSC.g':   26.80,
    'Subaru_HSC.z':   25.90,
    'CTIO_DECam.g':   26.46,
    'CTIO_DECam.r':   25.73,
    'CTIO_DECam.i':   25.54,
    'CTIO_DECam.z':   24.97,
}

# Exact order from filters_to_use.dat — must not be changed
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

SHORT_NAMES = ['NISP-H', 'NISP-J', 'NISP-Y', 'VIS', 'HSC-g', 'HSC-z',
               'DECam-g', 'DECam-r', 'DECam-i', 'DECam-z']

# All three photometry types share the same survey-depth-based 1σ limits
PHOT_TYPES = ['2fwhm', '3fwhm', 'templfit']
HEMISPHERE  = 'north'


def mag_to_flux_ujy(m):
    """AB magnitude → flux in μJy."""
    return 3631.0e6 * 10.0 ** (-0.4 * m)


def main():
    # Resolve output directory relative to this file so the script works
    # regardless of cwd
    base = Path(__file__).resolve().parent

    m5_arr  = np.array([depth5_mag[k] for k in order], dtype=float)
    f5_ujy  = np.array([mag_to_flux_ujy(m) for m in m5_arr], dtype=float)
    f1_ujy  = f5_ujy / 5.0   # 1σ limit stored in background_noise files

    print('=' * 72)
    print('5σ DEPTH → 1σ FLUX LIMIT CONVERSION')
    print('=' * 72)
    print(f'{"Filter":>12}  {"m_5σ":>6}  {"f_5σ (μJy)":>12}  {"f_1σ (μJy)":>12}')
    print('-' * 50)
    for name, m5, f5, f1 in zip(SHORT_NAMES, m5_arr, f5_ujy, f1_ujy):
        print(f'{name:>12}  {m5:>6.2f}  {f5:>12.6f}  {f1:>12.6f}')

    # Load old values if they exist (for comparison)
    old_path = base / f'background_noise_{HEMISPHERE}_templfit.npy'
    if old_path.exists():
        old = np.load(old_path)
        print()
        print('Comparison with old empirical limits (templfit):')
        print(f'{"Filter":>12}  {"old (μJy)":>12}  {"new (μJy)":>12}  {"ratio":>8}')
        print('-' * 50)
        for name, o, n in zip(SHORT_NAMES, old, f1_ujy):
            ratio = n / o if o > 0 else float('nan')
            print(f'{name:>12}  {o:>12.6f}  {n:>12.6f}  {ratio:>8.2f}x')

    print()
    saved = []
    for phot_type in PHOT_TYPES:
        out = base / f'background_noise_{HEMISPHERE}_{phot_type}.npy'
        np.save(out, f1_ujy)
        saved.append(out)
        print(f'  Saved: {out.name}')

    print(f'\nDone — {len(saved)} files written.')
    print('Re-run learn_obs_noise_from_survey.py is NOT needed;')
    print('these files are used directly as sigma_lim in the noise model.')


if __name__ == '__main__':
    main()
