"""
Convenience launcher for the depth-corrected observational-noise validation.

This wraps `validate_noise_model.py` with recommended defaults for the
new flux-space depth-corrected noise model.

Example:
    python examples/validate_noise_model_obs_noise.py --n-sim 10000
"""

import sys
from validate_noise_model import main as _main


DEFAULTS = {
    "--noise-model": "depth_corrected",
    "--noise-prefix": "north_2fwhm",
    "--aperture": "2fwhm",
    "--std-scale": "1.15",
    "--smooth-bins": None,
    "--depth-nsigma": "1.0",
    "--corr-clip-min": "0.2",
    "--corr-clip-max": "5.0",
    "--selection-band": "VIS",
    "--mag-min": "22",
    "--mag-max": "28",
}


def _inject_defaults(argv):
    existing = set(argv)
    out = list(argv)

    for key, value in DEFAULTS.items():
        if key in existing:
            continue
        if value is None:
            out.append(key)
        else:
            out.extend([key, value])
    return out


def main():
    argv = [sys.argv[0]] + _inject_defaults(sys.argv[1:])
    sys.argv = argv
    _main()


if __name__ == "__main__":
    main()
