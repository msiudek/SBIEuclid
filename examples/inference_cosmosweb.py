"""
Stellar mass estimation for real Euclid galaxies matched to COSMOS-Web.

Uses a trained mass_sfr model to infer logM* and logSFR from 10-band
photometry in the matched_euclid_cosmosweb.fits catalog and compares to
COSMOS-Web reference quantities.

Usage
-----
python examples/inference_cosmosweb.py \
    --n-gal 500 --snr-min 3 --n-bands-min 7 --n-samples 200 \
    --outdir sbi-logs/inference_cosmosweb 2>&1 | tee sbi-logs/inference_cosmosweb.log

Selection criteria applied before inference:
  - valid z_lephare photometric redshift
  - valid COSMOS-Web logM_lephare reference mass
  - at least --n-bands-min filters with SNR >= --snr-min
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
OBS_DIR    = ROOT / "obs" / "obs_properties"
LIB_DIR    = ROOT / "library"
CATALOG    = ROOT / "obs" / "obs_properties" / "COSMOS-Web" / "matched_euclid_cosmosweb.fits"
MODEL_NAME = "model_euclid_v1.0.pkl"

# Filter order from filters_to_use.dat (must match training order)
FILTER_STEMS = [
    "h",            # NISP-H
    "j",            # NISP-J
    "y",            # NISP-Y
    "vis",          # VIS
    "g_ext_hsc",    # HSC-g
    "z_ext_hsc",    # HSC-z
    "g_ext_decam",  # DECam-g
    "r_ext_decam",  # DECam-r
    "i_ext_decam",  # DECam-i
    "z_ext_decam",  # DECam-z
]
FILTER_NAMES = ["NISP-H", "NISP-J", "NISP-Y", "VIS",
                "HSC-g", "HSC-z", "DECam-g", "DECam-r", "DECam-i", "DECam-z"]
N_FILT = len(FILTER_STEMS)

# Keep reference-sample diagnostics inside the same parameter support used in training
TRAIN_LOGM_MIN = 4.0
TRAIN_LOGM_MAX = 13.0
TRAIN_LOGSFR_MIN = -4.0
TRAIN_LOGSFR_MAX = 3.0


# ── photometry column helper ───────────────────────────────────────────────
def build_phot_col(stem, phot_type, err=False):
    """Return flux/fluxerr column name for a filter stem and photometry type.

    phot_type='templfit': flux_{stem}_templfit, except VIS → flux_vis_psf.
    phot_type='2fwhm'/'3fwhm': flux_{stem}_{phot_type}_aper.
    """
    prefix = "fluxerr" if err else "flux"
    if phot_type == "templfit":
        return f"{prefix}_vis_psf" if stem == "vis" else f"{prefix}_{stem}_templfit"
    return f"{prefix}_{stem}_{phot_type}_aper"


def validate_requested_phot_type(cat, requested_phot_type):
    """Validate that all required columns for requested phot_type exist."""
    missing_flux = [
        build_phot_col(stem, requested_phot_type, err=False)
        for stem in FILTER_STEMS
        if build_phot_col(stem, requested_phot_type, err=False) not in cat.colnames
    ]
    missing_err = [
        build_phot_col(stem, requested_phot_type, err=True)
        for stem in FILTER_STEMS
        if build_phot_col(stem, requested_phot_type, err=True) not in cat.colnames
    ]

    if len(missing_flux) > 0:
        raise KeyError(
            f"Requested phot_type='{requested_phot_type}' is missing {len(missing_flux)}/{N_FILT} "
            f"required flux columns. Missing examples: {missing_flux[:4]}. "
            "No fallback is applied."
        )

    if len(missing_err) > 0:
        raise KeyError(
            f"Requested phot_type='{requested_phot_type}' is missing {len(missing_err)}/{N_FILT} "
            f"required fluxerr columns. Missing examples: {missing_err[:4]}. "
            "No fallback is applied."
        )


def parse_args():
    p = argparse.ArgumentParser(description="SBI mass inference on COSMOS-Web matched catalog")
    p.add_argument("--n-gal",       type=int,   default=500,
                   help="Number of galaxies to run inference on (default: 500)")
    p.add_argument("--snr-min",     type=float, default=3.0,
                   help="Minimum SNR per band to count as detected (default: 3)")
    p.add_argument("--n-bands-min", type=int,   default=7,
                   help="Minimum number of bands with SNR >= snr-min (default: 7)")
    p.add_argument("--n-samples",   type=int,   default=200,
                   help="Number of posterior samples per galaxy (default: 200)")
    p.add_argument("--outdir",      type=str,   default="sbi-logs/inference_cosmosweb",
                   help="Output directory for results and plots")
    p.add_argument("--model-name",  type=str,   default=MODEL_NAME,
                   help=f"Model filename in library/ (default: {MODEL_NAME})")
    p.add_argument("--filter-list", type=str,   default="filters_to_use.dat",
                   help="Filter list file; bands are derived from it (default: filters_to_use.dat; "
                        "use filters_to_use_euclid_jwstnir.dat for Euclid+JWST-NIR)")
    p.add_argument("--noise-prefix", type=str,  default=None,
                   help="Custom noise model prefix. If unset, derived as 'north_{phot_type}'")
    p.add_argument("--catalog",     type=str,   default=None,
                   help="Override input catalog path (default: matched_euclid_cosmosweb.fits)")
    p.add_argument("--sample-with", type=str, default="rejection",
                   choices=["rejection", "mcmc", "direct"],
                   help="Posterior sampling backend (default: rejection). 'direct' "
                        "samples straight from the NPE flow (fast, no rejection/leakage "
                        "correction) and avoids the 0%%-acceptance stall seen with "
                        "'rejection' on this v3 model.")
    p.add_argument("--z-max-domain", type=float, default=6.0,
                   help="Domain pre-cut: drop galaxies with reference z above this. The v3 "
                        "model was trained to z_max~6; beyond it the NPE flow extrapolates, "
                        "putting posterior mass outside the prior -> 0%% sampling acceptance / stalls.")
    p.add_argument("--logm-max-domain", type=float, default=11.0,
                   help="Domain pre-cut: drop galaxies with reference logM above this. The v3 "
                        "prior caps logM at 11.39 and the model over-estimates mass, so "
                        "bright/massive galaxies leak past the ceiling and stall sampling.")
    p.add_argument("--phot-type",   type=str, default="templfit",
                   choices=["templfit", "2fwhm", "3fwhm"],
                   help=("Photometry type: 'templfit' (template-fit; VIS uses psf), "
                         "'2fwhm', or '3fwhm' aperture. Default: templfit"))
    p.add_argument("--observation-space", type=str, default="flux",
                   choices=["mag", "flux"],
                   help=(
                       "Feature space expected by the model: 'flux' for flux+sigma_flux "
                       "(keeps negative noisy realizations), 'mag' for legacy mag+sigma. "
                       "Default: flux"
                   ))
    p.add_argument("--device",      type=str,   default="cpu",
                   help="Inference device: cpu or cuda (default: cpu)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for galaxy selection (default: 42)")
    p.add_argument("--infer-snr-threshold", type=float, default=3.0,
                   help=(
                       "Inference-side SNR detection threshold used when building obs vectors "
                       "from flux/fluxerr (default: 3.0)."
                   ))
    p.add_argument("--snr-threshold-sweep", type=float, nargs="*", default=[],
                   help=(
                       "Optional extra inference-side thresholds for no-retraining bias test, "
                       "e.g. --snr-threshold-sweep 3 5 10. Uses same selected galaxies and model."
                   ))
    return p.parse_args()


def _find_first_existing_column(cat, candidates):
    for name in candidates:
        if name in cat.colnames:
            return name
    return None


# ── photometry helpers ─────────────────────────────────────────────────────

def build_obs_array(flux_2d, fluxerr_2d, limits, snr_threshold=1.0, observation_space="mag"):
    """
        Build the interleaved observation vector expected by the sbipix model.

        observation_space='mag':
            [mag_0, sig_mag_0, ..., mag_9, sig_mag_9], with mag=99 for non-detections.
        observation_space='flux':
            [flux_0, sig_flux_0, ..., flux_9, sig_flux_9], keeping full flux distribution.
    """
    n_gal = flux_2d.shape[0]
    obs = np.zeros((n_gal, 2 * N_FILT), dtype=np.float32)

    for j, lim in enumerate(limits):
        f   = flux_2d[:, j]
        fe  = fluxerr_2d[:, j]
        if observation_space == "flux":
            flux_obs = np.asarray(f, dtype=float)
            sigma_flux = np.asarray(fe, dtype=float)

            bad_flux = ~np.isfinite(flux_obs)
            flux_obs[bad_flux] = 0.0

            bad_sigma = ~np.isfinite(sigma_flux) | (sigma_flux <= 0)
            sigma_flux[bad_sigma] = max(float(lim), 1e-12)

            obs[:, 2 * j] = flux_obs
            obs[:, 2 * j + 1] = sigma_flux
            continue

        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.abs(f / np.where(fe > 0, fe, np.nan))
        snr_non_detect = ~np.isfinite(snr) | (snr < snr_threshold)
        non_detect = ~np.isfinite(f) | (f <= 0) | (f < lim) | snr_non_detect

        # magnitude
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.where(non_detect, 99.0, -2.5 * np.log10(f / 3631e6))

        # magnitude sigma  (2.5/ln10 * sigma_flux/flux for detections,
        #                   limiting mag for non-detections)
        lim_mag = -2.5 * np.log10(lim / 3631e6) if lim > 0 else 99.0
        with np.errstate(divide='ignore', invalid='ignore'):
            sig = np.where(non_detect,
                           lim_mag,
                           (2.5 / np.log(10)) * np.abs(fe / f))
        sig = np.clip(sig, 0.001, 5.0)

        obs[:, 2 * j]     = mag
        obs[:, 2 * j + 1] = sig

    return obs


def run_inference_at_threshold(sx, qphi, flux_sel, fluxerr_sel, limits, z_sel, args, snr_threshold):
    """Run posterior inference for one inference-side SNR threshold."""
    obs = build_obs_array(
        flux_sel,
        fluxerr_sel,
        limits,
        snr_threshold=snr_threshold,
        observation_space=args.observation_space,
    )

    if args.observation_space == "flux":
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.abs(obs[:, ::2] / np.where(obs[:, 1::2] > 0, obs[:, 1::2], np.nan))
        detection_fraction = float(np.mean(np.isfinite(snr) & (snr >= snr_threshold)))
    else:
        detection_fraction = float(np.mean(obs[:, ::2] < 99.0))

    posteriors = sx._get_posterior_obs(
        obs,
        qphi,
        n_samples=args.n_samples,
        bar=True,
        input_z=z_sel,
        device=args.device,
        sample_with=args.sample_with,
    )

    logM_med = np.nanmedian(posteriors[:, :, 0], axis=1)
    logM_lo = np.nanpercentile(posteriors[:, :, 0], 16, axis=1)
    logM_hi = np.nanpercentile(posteriors[:, :, 0], 84, axis=1)
    logSFR_med = np.nanmedian(posteriors[:, :, 1], axis=1)

    return {
        "obs": obs,
        "detection_fraction": detection_fraction,
        "posteriors": posteriors,
        "logM_med": logM_med,
        "logM_lo": logM_lo,
        "logM_hi": logM_hi,
        "logSFR_med": logSFR_med,
    }


# ── main ──────────────────────────────────────────────────────────────────
def main():
    global FILTER_STEMS, FILTER_NAMES, N_FILT
    from sbipix import sbipix
    from sbipix.utils import validation_plots as vplots
    from sbipix.utils.sed_utils import load_filter_metadata

    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    # Derive bands from the requested filter list (so 12-band Euclid+JWST works)
    _meta = load_filter_metadata(args.filter_list, filt_dir=str(OBS_DIR))
    FILTER_STEMS = [m["col_stem"] for m in _meta]
    FILTER_NAMES = [m["short"]    for m in _meta]
    N_FILT = len(FILTER_STEMS)
    print(f"Loaded {N_FILT} filters from {args.filter_list}: {', '.join(FILTER_NAMES)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    requested_phot_type = args.phot_type

    # ------------------------------------------------------------------
    # 1. Load catalog and apply selection
    # ------------------------------------------------------------------
    catalog_path = args.catalog if args.catalog else str(CATALOG)
    print(f"Loading catalog: {catalog_path}")
    cat = Table.read(catalog_path)
    N_total = len(cat)
    print(f"  {N_total} total matched galaxies")

    validate_requested_phot_type(cat, requested_phot_type)
    effective_phot_type = requested_phot_type
    noise_prefix = args.noise_prefix if args.noise_prefix else f"north_{effective_phot_type}"
    print(f"Photometry type: {effective_phot_type}  (noise prefix: {noise_prefix})")

    # Build flux and fluxerr arrays (n_gal, n_filt) in μJy
    try:
        flux = np.column_stack([
            np.array(cat[build_phot_col(stem, effective_phot_type, err=False)], dtype=float)
            for stem in FILTER_STEMS
        ])
        fluxerr = np.column_stack([
            np.array(cat[build_phot_col(stem, effective_phot_type, err=True)], dtype=float)
            for stem in FILTER_STEMS
        ])
    except KeyError as exc:
        raise KeyError(
            f"Column {exc} not found in catalog for phot_type='{effective_phot_type}'. "
            "For 'templfit' the matched catalog must include templfit/psf columns "
            "(re-run the catalog matching script to add them)."
        ) from exc

    # Reference values (matched_euclid_cosmosweb.fits: LePhare + z_lephare)
    z_ref    = np.array(cat["z_lephare"],        dtype=float)
    mass_ref = np.array(cat["logM_lephare"],     dtype=float)   # log10(M/Msun)
    mass_lo  = np.array(cat["logM_l68_lephare"], dtype=float)
    mass_hi  = np.array(cat["logM_u68_lephare"], dtype=float)

    # SFR reference columns (logSFR_lephare is log10 M☉/yr)
    sfr_ref_col = _find_first_existing_column(
        cat, ["logSFR_lephare", "sfr_med", "logsfr_med", "sfr", "log_sfr"])
    sfr_lo_col  = _find_first_existing_column(
        cat, ["logSFR_l68_lephare", "sfr_l68", "logsfr_l68", "sfr_lo", "log_sfr_lo"])
    sfr_hi_col  = _find_first_existing_column(
        cat, ["logSFR_u68_lephare", "sfr_u68", "logsfr_u68", "sfr_hi", "log_sfr_hi"])

    if sfr_ref_col is not None:
        sfr_ref = np.array(cat[sfr_ref_col], dtype=float)
        sfr_lo = np.array(cat[sfr_lo_col], dtype=float) if sfr_lo_col is not None else np.full(len(cat), np.nan)
        sfr_hi = np.array(cat[sfr_hi_col], dtype=float) if sfr_hi_col is not None else np.full(len(cat), np.nan)
        print(
            f"Using COSMOS-Web SFR reference columns: med='{sfr_ref_col}', "
            f"lo='{sfr_lo_col}', hi='{sfr_hi_col}'"
        )
    else:
        sfr_ref = np.full(len(cat), np.nan)
        sfr_lo = np.full(len(cat), np.nan)
        sfr_hi = np.full(len(cat), np.nan)
        print("No COSMOS-Web SFR reference columns found; SFR comparison plot will be skipped")

    # Per-band SNR
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(flux / np.where(fluxerr > 0, fluxerr, np.nan))

    n_bands_snr = np.sum((snr >= args.snr_min) & np.isfinite(snr), axis=1)

    # Selection mask (aligned with training support ranges)
    # Base cuts (finite + broad training ranges), then a DOMAIN pre-cut to the
    # region where the v3 flow does NOT leak outside its prior box (avoids the
    # 0% sampling-acceptance stalls on out-of-domain bright/massive/high-z gal).
    has_z    = np.isfinite(z_ref)  & (z_ref > 0)
    has_mass = np.isfinite(mass_ref) & (mass_ref > TRAIN_LOGM_MIN) & (mass_ref < TRAIN_LOGM_MAX)
    has_sfr = np.ones(len(cat), dtype=bool)
    if sfr_ref_col is not None:
        has_sfr = np.isfinite(sfr_ref) & (sfr_ref > TRAIN_LOGSFR_MIN) & (sfr_ref < TRAIN_LOGSFR_MAX)
    has_bands = n_bands_snr >= args.n_bands_min

    in_domain = (z_ref <= args.z_max_domain) & (mass_ref <= args.logm_max_domain)
    n_ood_z = int((has_z & (z_ref > args.z_max_domain)).sum())
    n_ood_m = int((has_mass & (mass_ref > args.logm_max_domain)).sum())
    print(f"  domain pre-cut (v3 support): excluded {n_ood_z} gal z>{args.z_max_domain:g}, "
          f"{n_ood_m} gal logM>{args.logm_max_domain:g} (would leak / stall sampling)")

    good = has_z & has_mass & has_sfr & has_bands & in_domain

    if sfr_ref_col is not None:
        print(
            f"  {good.sum()} galaxies pass: z valid + "
            f"{TRAIN_LOGM_MIN:.0f}<logM<{TRAIN_LOGM_MAX:.0f} + "
            f"{TRAIN_LOGSFR_MIN:.0f}<logSFR<{TRAIN_LOGSFR_MAX:.0f} + "
            f"≥{args.n_bands_min} bands SNR≥{args.snr_min}"
        )
    else:
        print(
            f"  {good.sum()} galaxies pass: z valid + "
            f"{TRAIN_LOGM_MIN:.0f}<logM<{TRAIN_LOGM_MAX:.0f} + "
            f"≥{args.n_bands_min} bands SNR≥{args.snr_min}"
        )

    good_idx = np.where(good)[0]
    if len(good_idx) < args.n_gal:
        print(f"  WARNING: only {len(good_idx)} galaxies pass cuts; using all of them")
        sel = good_idx
    else:
        sel = rng.choice(good_idx, size=args.n_gal, replace=False)
    sel = np.sort(sel)
    print(f"  Selected {len(sel)} galaxies for inference")

    flux_sel    = flux[sel]
    fluxerr_sel = fluxerr[sel]
    z_sel       = z_ref[sel]
    mass_sel    = mass_ref[sel]
    mass_lo_sel = mass_lo[sel]
    mass_hi_sel = mass_hi[sel]
    sfr_sel     = sfr_ref[sel]
    sfr_lo_sel  = sfr_lo[sel]
    sfr_hi_sel  = sfr_hi[sel]
    nbands_sel  = n_bands_snr[sel]

    # ------------------------------------------------------------------
    # 2. Configure sbipix model
    # ------------------------------------------------------------------
    print("\nConfiguring sbipix model...")
    sx = sbipix()
    sx.configure_filters(
        filter_list=args.filter_list,
        filter_path=str(OBS_DIR),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )
    sx.model_path = str(LIB_DIR) + "/"
    sx.model_name = args.model_name
    sx.infer_z    = False  # we want to condition on catalog redshift, not infer z
    sx.include_limit   = True
    sx.include_sigma   = True
    sx.condition_sigma = True
    sx.configure_noise_model(
        sigma_sampler="mag_lognormal",
        detection_model="hard",
        observation_space=args.observation_space,
    )
    sx.load_obs_features()    # populates sx.limits, sx.mean_sigma_obs, sx.percentiles, etc.
    limits = sx.limits        # (n_filt,) in μJy
    print(f"  Flux limits (μJy): {np.array(limits).round(6)}")

    # ------------------------------------------------------------------
    # 3. Build observation array (n_gal, 2*n_filt) - correct mag+magerr
    # ------------------------------------------------------------------
    print("Building observation array...")
    obs = build_obs_array(
        flux_sel,
        fluxerr_sel,
        limits,
        snr_threshold=args.infer_snr_threshold,
        observation_space=args.observation_space,
    )
    print(f"  obs shape: {obs.shape}  ({len(sel)} galaxies, {2*N_FILT} features)")
    print(f"  observation space: {args.observation_space}")
    print(f"  inference SNR threshold: {args.infer_snr_threshold}")
    if args.observation_space == "flux":
        with np.errstate(divide='ignore', invalid='ignore'):
            snr_dbg = np.abs(obs[:, ::2] / np.where(obs[:, 1::2] > 0, obs[:, 1::2], np.nan))
        print(f"  detection fraction (SNR>={args.infer_snr_threshold:g}): {np.mean(np.isfinite(snr_dbg) & (snr_dbg >= args.infer_snr_threshold)):.3f}")
    else:
        print(f"  detection fraction (<99 mag): {np.mean(obs[:, ::2] < 99.0):.3f}")

    # Quick sanity: print first galaxy
    if args.observation_space == "flux":
        print(f"  galaxy[0] flux : {obs[0, ::2].round(6)}")
        print(f"  galaxy[0] sigF : {obs[0, 1::2].round(6)}")
    else:
        print(f"  galaxy[0] mags : {obs[0, ::2].round(2)}")
        print(f"  galaxy[0] sigs : {obs[0, 1::2].round(3)}")

    # ------------------------------------------------------------------
    # 4. Load model and run inference
    # ------------------------------------------------------------------
    model_file = sx.model_path + sx.model_name
    anpe_file = sx.model_path + "anpe_" + sx.model_name
    print(f"\nLoading model: {model_file}")

    qphi = None
    expected_ctx = obs.shape[1] + 1
    probe_errors = []

    def _supports_obs_plus_z(posterior):
        try:
            posterior.sample(
                (1,),
                x=torch.zeros((1, expected_ctx), dtype=torch.float32, device=args.device),
                show_progress_bars=False,
            )
            return True, None
        except Exception as exc:
            return False, str(exc)

    # Helper: load pickle file remapping any CUDA tensors to target device.
    # torch.load with map_location doesn't reach tensors nested inside pickle
    # objects, so we temporarily patch torch.load itself.
    import functools as _functools
    def _safe_load(path):
        _orig = torch.load
        try:
            torch.load = _functools.partial(_orig, map_location=args.device, weights_only=False)
            with open(path, "rb") as _f:
                return pickle.load(_f)
        finally:
            torch.load = _orig

    # Prefer rebuilding posterior from SNPE object when available.
    try:
        anpe = _safe_load(anpe_file)
        try:
            qphi = anpe.build_posterior(sample_with=args.sample_with)
            print(f"Using posterior sampler backend: {args.sample_with}")
        except TypeError:
            qphi = anpe.build_posterior()
            print(
                "WARNING: current sbi version does not support "
                "build_posterior(sample_with=...). Using default posterior backend."
            )

        ok, err = _supports_obs_plus_z(qphi)
        if not ok:
            probe_errors.append(f"anpe posterior incompatible with obs+z context ({err})")
            qphi = None
    except Exception as exc:
        print(
            f"WARNING: could not load/rebuild posterior from {anpe_file} ({exc}). "
            "Falling back to pickled posterior object."
        )
        probe_errors.append(f"anpe load/rebuild failed ({exc})")

    if qphi is None:
        qphi_model = _safe_load(model_file)

        ok, err = _supports_obs_plus_z(qphi_model)
        if ok:
            qphi = qphi_model
            print("Using pickled posterior object from model file.")
        else:
            probe_errors.append(f"model posterior incompatible with obs+z context ({err})")

    # Ensure at least one candidate posterior supports conditioning on catalog redshift.
    if qphi is None:
        detail = " | ".join(probe_errors) if len(probe_errors) > 0 else "no probe details available"
        raise RuntimeError(
            "Selected model is incompatible with catalog-redshift conditioning (obs+z input). "
            "This posterior expects photometry-only context. "
            "Please use/retrain a model trained with --z-mode condition (infer_z=False), "
            "and ensure the matching anpe_* model artifact is available. "
            f"Model: {sx.model_name}; expected context in this script: {expected_ctx}; "
            f"checked anpe file: {anpe_file}; probe details: {detail}."
        )

    print(
        f"Running inference on {len(sel)} galaxies × {args.n_samples} samples "
        f"(backend={args.sample_with}, snr_threshold={args.infer_snr_threshold}) ..."
    )
    baseline = run_inference_at_threshold(
        sx, qphi, flux_sel, fluxerr_sel, limits, z_sel, args, args.infer_snr_threshold
    )
    posteriors = baseline["posteriors"]

    # ------------------------------------------------------------------
    # 5. Extract summary statistics
    # ------------------------------------------------------------------
    # theta order for mass_sfr: [0]=logM*, [1]=logSFR
    logM_med = baseline["logM_med"]
    logM_lo = baseline["logM_lo"]
    logM_hi = baseline["logM_hi"]
    logSFR_med = baseline["logSFR_med"]

    # Optional no-retraining threshold sweep diagnostics
    if len(args.snr_threshold_sweep) > 0:
        sweep_thresholds = [float(args.infer_snr_threshold)]
        for value in args.snr_threshold_sweep:
            threshold = float(value)
            if threshold not in sweep_thresholds:
                sweep_thresholds.append(threshold)

        print("\nNo-retraining SNR-threshold sweep diagnostics:")
        print(f"  baseline threshold={args.infer_snr_threshold:.2f}")

        detection_fractions = []
        delta_logm_stack = []

        for threshold in sweep_thresholds:
            run = baseline if threshold == float(args.infer_snr_threshold) else run_inference_at_threshold(
                sx, qphi, flux_sel, fluxerr_sel, limits, z_sel, args, threshold
            )
            detection_fractions.append(run["detection_fraction"])

            delta_logm = run["logM_med"] - baseline["logM_med"]
            delta_logm_stack.append(delta_logm)

            p16, p50, p84 = np.nanpercentile(delta_logm, [16, 50, 84])
            print(
                f"  threshold={threshold:>4.1f}: "
                f"det_frac={run['detection_fraction']:.3f}, "
                f"ΔlogM* median={p50:+.3f} dex "
                f"(p16={p16:+.3f}, p84={p84:+.3f})"
            )

        sweep_file = outdir / "snr_threshold_sweep.npz"
        np.savez(
            sweep_file,
            thresholds=np.asarray(sweep_thresholds, dtype=float),
            detection_fraction=np.asarray(detection_fractions, dtype=float),
            delta_logM=np.asarray(delta_logm_stack, dtype=float),
            baseline_logM=baseline["logM_med"],
            selected_indices=sel,
        )
        print(f"  Sweep saved to {sweep_file}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    result_file = outdir / "inference_results.npz"
    np.savez(result_file,
             logM_sbi=logM_med, logM_sbi_lo=logM_lo, logM_sbi_hi=logM_hi,
             logSFR_sbi=logSFR_med,
             logM_cosmosweb=mass_sel,
             logM_cosmosweb_lo=mass_lo_sel,
             logM_cosmosweb_hi=mass_hi_sel,
             logSFR_cosmosweb=sfr_sel,
             logSFR_cosmosweb_lo=sfr_lo_sel,
             logSFR_cosmosweb_hi=sfr_hi_sel,
             z=z_sel, n_bands=nbands_sel,
             posteriors=posteriors,
             selected_indices=sel)
    print(f"\nResults saved to {result_file}")

    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    valid = np.isfinite(logM_med) & np.isfinite(mass_sel)
    delta = logM_med[valid] - mass_sel[valid]
    r, _ = pearsonr(mass_sel[valid], logM_med[valid])
    print(f"\nMass comparison (N={valid.sum()}):")
    print(f"  Pearson r              = {r:.3f}")
    print(f"  Median Δ(SBI-CWeb)     = {np.median(delta):.3f} dex")
    print(f"  NMAD(Δ)                = {1.4826 * np.median(np.abs(delta - np.median(delta))):.3f} dex")
    print(f"  Std(Δ)                 = {np.std(delta):.3f} dex")
    print(f"  SBI logM* range        : [{logM_med[valid].min():.2f}, {logM_med[valid].max():.2f}]")
    print(f"  CWeb logM (LePhare)    : [{mass_sel[valid].min():.2f}, {mass_sel[valid].max():.2f}]")

    sfr_valid = np.isfinite(logSFR_med) & np.isfinite(sfr_sel)
    if np.any(sfr_valid):
        sfr_delta = logSFR_med[sfr_valid] - sfr_sel[sfr_valid]
        print(f"\nSFR comparison (N={sfr_valid.sum()}):")
        print(f"  Median Δ(SBI-CWeb)     = {np.median(sfr_delta):.3f} dex")
        print(f"  NMAD(Δ)                = {1.4826 * np.median(np.abs(sfr_delta - np.median(sfr_delta))):.3f} dex")
        print(f"  Std(Δ)                 = {np.std(sfr_delta):.3f} dex")
        print(f"  SBI logSFR range       : [{logSFR_med[sfr_valid].min():.2f}, {logSFR_med[sfr_valid].max():.2f}]")
        print(f"  CWeb logSFR (LePhare)  : [{sfr_sel[sfr_valid].min():.2f}, {sfr_sel[sfr_valid].max():.2f}]")

    # ------------------------------------------------------------------
    # 8. Plots (shared validation plotting utilities)
    # ------------------------------------------------------------------
    plot_file = vplots.plot_mass_comparison(mass_sel[valid], logM_med[valid], z_sel[valid], outdir)
    if plot_file is not None:
        print(f"Plot saved to {plot_file}")

    plot_file2 = vplots.plot_posterior_width_vs_mass(mass_sel[valid], logM_lo[valid], logM_hi[valid], z_sel[valid], outdir)
    if plot_file2 is not None:
        print(f"Plot saved to {plot_file2}")

    plot_file3 = vplots.plot_sfr_mass(logM_med[valid], logSFR_med[valid], z_sel[valid], outdir)
    if plot_file3 is not None:
        print(f"Plot saved to {plot_file3}")

    plot_file4 = vplots.plot_sfr_comparison(sfr_sel, logSFR_med, z_sel, outdir)
    if plot_file4 is not None:
        print(f"Plot saved to {plot_file4}")

    print("\nDone.")


if __name__ == "__main__":
    main()
