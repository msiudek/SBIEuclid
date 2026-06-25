"""
JADES baseline (ladder rung 0): recreate the paper's parametric τ-delayed model
with the CLEAN upstream SBIPIX code (patriglesias/SBIPIX @247055e).

This mirrors examples/upstream_simulation_training_testing.py but:
  - parametric=True (τ-delayed SFH) + both_masses=True  → the paper's shipped
    "tau-delayed, 100k, z<7.5" configuration (cf. upstream_inference_six_gal.py),
  - 100k simulations (not 1M),
  - the train block is active (not commented),
  - everything is argparse-driven.

No fork modifications: SFH priors, FSPS config, training are exactly upstream.
Uses the JADES pixel noise model (mean/std/percentiles_jades_res_bins.npy,
background_noise_hainline.npy) that ships with the paper.

The atlas stores noiseless FSPS SEDs, so it is noise-independent and can be
REUSED later (retrain only) for the integrated/Euclid rungs.

Run on the server (needs FSPS):
    python examples/jades_train.py \
        --n-simulation 100000 \
        --atlas-name atlas_obs_jades_100k \
        --model-name model_jades_100k.pkl \
        --device cuda 2>&1 | tee sbi-logs/jades_train.log
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = ROOT / "obs" / "obs_properties"
LIB_DIR = ROOT / "library"


def parse_args():
    p = argparse.ArgumentParser(description="Recreate JADES parametric model (clean upstream)")
    p.add_argument("--n-simulation", type=int, default=100000)
    p.add_argument("--atlas-name", type=str, default="atlas_obs_jades_100k")
    p.add_argument("--model-name", type=str, default="model_jades_100k.pkl")
    p.add_argument("--filter-list", type=str, default="filters_jades_no_wfc.dat")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-simulate", action="store_true",
                   help="Reuse an existing atlas (atlas is noise-independent)")
    # paper priors (upstream README / inference_six_gal model)
    p.add_argument("--mass-min", type=float, default=4.0)
    p.add_argument("--mass-max", type=float, default=12.0)
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=7.5)
    p.add_argument("--Z-min", type=float, default=-2.27)
    p.add_argument("--Z-max", type=float, default=0.4)
    p.add_argument("--Av-min", type=float, default=0.0)
    p.add_argument("--Av-max", type=float, default=4.0)
    # training
    p.add_argument("--nblocks", type=int, default=5)
    p.add_argument("--nhidden", type=int, default=128)
    p.add_argument("--epochs-max", type=int, default=None)
    p.add_argument("--n-test", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    from sbipix import sbipix
    from sbipix.plotting.diagnostics import plot_test_performance

    sx = sbipix()
    sx.filter_path = str(OBS_DIR) + "/"
    sx.filter_list = args.filter_list
    sx.atlas_path = str(LIB_DIR) + "/"
    sx.atlas_name = args.atlas_name
    sx.model_path = str(LIB_DIR) + "/"
    sx.model_name = args.model_name
    sx.n_simulation = args.n_simulation

    # Paper's shipped parametric configuration
    sx.parametric = True       # τ-delayed SFH
    sx.both_masses = True       # report M* and M*_formed
    sx.infer_z = False          # condition on redshift, do not infer it

    # 1. Simulate atlas (noiseless FSPS SEDs) — paper priors
    if not args.skip_simulate:
        print(f"Simulating {args.n_simulation} τ-delayed SEDs → {args.atlas_name}")
        sx.simulate(
            mass_min=args.mass_min, mass_max=args.mass_max,
            z_prior="flat", z_min=args.z_min, z_max=args.z_max,
            Z_min=args.Z_min, Z_max=args.Z_max,
            dust_model="Calzetti", dust_prior="flat",
            Av_min=args.Av_min, Av_max=args.Av_max,
        )

    # 2. Load atlas
    sx.load_simulation()

    # 3. Observational realism (JADES pixel noise model)
    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True
    sx.load_obs_features()
    sx.add_noise_nan_limit_all()

    # 4. Clean non-finite parameter rows
    sim_ok = np.isfinite(np.sum(sx.theta, axis=1))
    sx.theta = sx.theta[sim_ok, :]
    sx.mag = sx.mag[sim_ok, :, :]
    sx.obs = sx.obs[sim_ok, :]
    sx.n_simulation = len(sx.theta[:, 0])
    print(f"Training on {sx.n_simulation} clean simulations")

    # 5. Parameter bounds from data (auto-sizes to the parameter vector)
    min_thetas = np.min(sx.theta, axis=0)
    max_thetas = np.max(sx.theta, axis=0)
    print("min_thetas:", np.round(min_thetas, 3))
    print("max_thetas:", np.round(max_thetas, 3))

    # 6. Train the normalizing flow
    sx.train(
        min_thetas=min_thetas, max_thetas=max_thetas,
        n_max=len(sx.theta), nblocks=args.nblocks, nhidden=args.nhidden,
        epochs_max=args.epochs_max, device=args.device,
    )

    # 7. Diagnostics
    sx.test_performance(n_test=args.n_test, return_posterior=True, device=args.device)
    try:
        plot_test_performance(sx, n_test=args.n_test)
    except Exception as exc:
        print(f"(plot_test_performance skipped: {exc})")
    print("Done.")


if __name__ == "__main__":
    main()
