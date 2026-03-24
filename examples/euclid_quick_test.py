from pathlib import Path
import argparse
import os
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quick SBIPIX Euclid smoke test (simulate + realism + optional train/test)"
    )
    parser.add_argument("--n-sim", type=int, default=300,
                        help="Number of simulations for quick run (default: 300)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Max epochs for quick training (default: 5)")
    parser.add_argument("--n-test", type=int, default=10,
                        help="Number of test samples for performance check (default: 10)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Posterior samples per test object (default: 100)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for train/test (default: cpu)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Stop after simulation + observational realism")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip performance test after training")
    parser.add_argument("--no-plot-test", action="store_true",
                        help="Skip the test-performance diagnostic plots")
    return parser


def main():
    args = build_parser().parse_args()

    if not os.environ.get("DISPLAY") and "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "Agg"

    from sbipix import sbipix

    project_root = Path(__file__).resolve().parents[1]
    obs_dir = project_root / "obs" / "obs_properties"
    library_dir = project_root / "library"
    library_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    sx = sbipix()

    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(obs_dir),
        mean_sigma_file="mean_sigma_north_3fwhm.npy",
        std_sigma_file="std_sigma_north_3fwhm.npy",
        percentiles_file="percentiles_north_3fwhm.npy",
        limits_file="background_noise_north_3fwhm.npy",
        lam_eff_file="lam_eff_north_3fwhm.npy",
    )

    sx.atlas_path = str(library_dir) + "/"
    sx.model_path = str(library_dir) + "/"
    sx.atlas_name = "atlas_obs_euclid_north_quick"
    sx.model_name = "post_obs_euclid_north_quick.pkl"

    sx.n_simulation = args.n_sim
    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False

    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True

    print("[1/5] Simulating galaxy SEDs...")
    sx.simulate(
        mass_min=6.0,
        mass_max=11.5,
        z_prior="flat",
        z_min=0.1,
        z_max=3.0,
        Z_min=-1.7,
        Z_max=0.3,
        dust_model="Calzetti",
        dust_prior="flat",
        Av_min=0.0,
        Av_max=2.5,
    )

    print("[2/5] Loading simulation...")
    sx.load_simulation()

    print("[3/5] Loading observational realism and adding noise/limits...")
    sx.load_obs_features()
    sx.add_noise_nan_limit_all()

    sim_ok = np.isfinite(np.sum(sx.theta, axis=1))
    sx.theta = sx.theta[sim_ok, :]
    sx.mag = sx.mag[sim_ok, :, :]
    sx.obs = sx.obs[sim_ok, :]
    sx.n_simulation = len(sx.theta)
    print(f"    Kept {sx.n_simulation} valid simulations after cleaning")

    if args.skip_train:
        print("Done: quick Euclid preparation finished (simulation + realism).")
        print("Run without --skip-train to continue to training/testing.")
        return

    print("[4/5] Training quick model...")
    min_thetas = np.min(sx.theta, axis=0)
    max_thetas = np.max(sx.theta, axis=0)

    sx.train(
        min_thetas=min_thetas,
        max_thetas=max_thetas,
        n_max=len(sx.theta),
        epochs_max=args.epochs,
        nblocks=3,
        nhidden=64,
        val_fraction=0.1,
        device=args.device,
    )

    if args.skip_test:
        print("Done: training finished. Skipping test as requested.")
        return

    print("[5/5] Testing quick model performance...")
    posterior_test = sx.test_performance(
        n_test=min(args.n_test, len(sx.theta)),
        n_samples=args.n_samples,
        return_posterior=True,
        device=args.device,
    )
    print(f"Posterior test shape: {posterior_test.shape}")

    if not args.no_plot_test:
        from sbipix.plotting import plot_test_performance

        logs_dir = project_root / "sbi-logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        print("[6/6] Plotting test-performance diagnostics...")
        plot_test_performance(
            sx,
            n_test=min(posterior_test.shape[0], len(sx.means_test)),
            save=True,
            name="euclid_quick_test_",
        )
        print(f"Saved diagnostic plots in {logs_dir}")

    print("Done: quick Euclid smoke test complete.")


if __name__ == "__main__":
    main()
