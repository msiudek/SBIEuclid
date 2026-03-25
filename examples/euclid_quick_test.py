from pathlib import Path
import argparse
import os
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quick SBIPIX Euclid smoke test (simulate + realism + optional train/test)"
    )
    parser.add_argument("--n-sim", type=int, default=50000,
                        help="Number of simulations for run (default: 50000)")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Max epochs for training (default: 250)")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Number of test samples for performance check (default: 100)")
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
    parser.add_argument("--skip-simulate", action="store_true",
                        help="Skip simulation and realism; load existing atlas directly")
    parser.add_argument("--atlas-name", default=None,
                        help="Override atlas name (default: auto-generated from noise-prefix). "
                             "Use to point to an existing atlas, e.g. atlas_obs_euclid_north_2fwhm_validate_10000")
    # Noise configuration options
    parser.add_argument("--noise-prefix", default="north_2fwhm",
                        choices=["north_2fwhm", "north_3fwhm"],
                        help="Noise prefix / aperture config (default: north_2fwhm)")
    parser.add_argument("--aperture", default="2fwhm",
                        choices=["2fwhm", "3fwhm"],
                        help="Aperture size (default: 2fwhm)")
    parser.add_argument("--noise-model", default="sigma_mag",
                        choices=["sigma_mag", "depth_corrected"],
                        help="Noise model: sigma(mag) or depth-corrected (default: sigma_mag)")
    parser.add_argument("--std-scale", type=float, default=1.2,
                        help="Standard deviation scale factor (default: 1.2)")
    parser.add_argument("--smooth-bins", action="store_true", default=True,
                        help="Interpolate sigma statistics between bins (default: True)")
    parser.add_argument("--detection-model", default="probabilistic",
                        choices=["hard", "probabilistic"],
                        help="Detection model (default: probabilistic)")
    parser.add_argument("--sigma-sampler", default="lognormal",
                        choices=["truncnorm", "lognormal"],
                        help="Sigma sampler distribution (default: lognormal)")
    parser.add_argument("--sigma-clip-max", type=float, default=0.8,
                        help="Clip sampled sigma above this mag threshold (default: 0.8)")
    parser.add_argument("--fits-file", default="obs/obs_properties/COSMOS_DEEP.fits",
                        help="Real COSMOS-Deep FITS file for mock matching")
    parser.add_argument("--patch-id", type=int, default=98,
                        help="Patch ID for real data matching (default: 98)")
    parser.add_argument("--mock-match", choices=["none", "vis1d", "vis_color2d"], default="vis1d",
                        help="Optional real-data matching for training mocks (default: vis1d)")
    parser.add_argument("--mock-match-band", default="VIS",
                        help="Band used for mock matching (default: VIS)")
    parser.add_argument("--mock-match-color-band", default="NISP-Y",
                        help="Color reference band for vis_color2d mode (default: NISP-Y)")
    parser.add_argument("--mock-match-bins", type=int, default=24,
                        help="Number of bins for mock matching histograms (default: 24)")
    parser.add_argument("--target-params", choices=["all", "no_tau_met"], default="no_tau_met",
                        help="Target parameter set for training (default: no_tau_met)")
    parser.add_argument("--theta-normalization", choices=["none", "zscore"], default="zscore",
                        help="Normalization for target parameters before training (default: zscore)")
    parser.add_argument("--sfr-floor", type=float, default=-5.0,
                        help="Lower clip for log(SFR) (default: -5)")
    parser.add_argument("--sfr-ceil", type=float, default=3.0,
                        help="Upper clip for log(SFR) (default: 3)")
    parser.add_argument("--max-train-samples", type=int, default=50000,
                        help="Maximum samples used for training (default: 50000)")
    parser.add_argument("--mass-min", type=float, default=6.0,
                        help="Simulation prior lower bound for stellar mass")
    parser.add_argument("--mass-max", type=float, default=11.5,
                        help="Simulation prior upper bound for stellar mass")
    parser.add_argument("--z-min", type=float, default=0.1,
                        help="Simulation prior lower bound for redshift")
    parser.add_argument("--z-max", type=float, default=3.0,
                        help="Simulation prior upper bound for redshift")
    parser.add_argument("--Av-min", type=float, default=0.0,
                        help="Simulation prior lower bound for dust Av")
    parser.add_argument("--Av-max", type=float, default=2.0,
                        help="Simulation prior upper bound for dust Av")
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
        mean_sigma_file=f"mean_sigma_{args.noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{args.noise_prefix}.npy",
        percentiles_file=f"percentiles_{args.noise_prefix}.npy",
        limits_file=f"background_noise_{args.noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{args.noise_prefix}.npy",
    )

    sx.atlas_path = str(library_dir) + "/"
    sx.model_path = str(library_dir) + "/"
    sx.atlas_name = args.atlas_name if args.atlas_name else f"atlas_obs_euclid_{args.noise_prefix}_quick"
    sx.model_name = f"post_obs_euclid_{args.noise_prefix}_quick.pkl"

    sx.n_simulation = args.n_sim
    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False

    sx.include_limit = True
    sx.condition_sigma = True
    sx.include_sigma = True

    # Configure noise model with optimal parameters
    sx.configure_noise_model(
        noise_model=args.noise_model,
        std_scale=args.std_scale,
        smooth_bins=args.smooth_bins,
        sigma_sampler=args.sigma_sampler,
        sigma_clip_max=args.sigma_clip_max,
        detection_model=args.detection_model,
    )

    print(f"[0/5] Configuration: {args.noise_prefix}, aperture={args.aperture}, "
          f"noise_model={args.noise_model}, std_scale={args.std_scale}, "
          f"detection={args.detection_model}, sampler={args.sigma_sampler}")
    print(f"      atlas: {sx.atlas_name}{' (reusing existing)' if args.skip_simulate else ''}")
    print()

    if args.skip_simulate:
        print("[1/5] Skipping simulation — loading existing atlas...")
    else:
        print("[1/5] Simulating galaxy SEDs...")
        sx.simulate(
            mass_min=args.mass_min,
            mass_max=args.mass_max,
            z_prior="flat",
            z_min=args.z_min,
            z_max=args.z_max,
            Z_min=-1.7,
            Z_max=0.3,
            dust_model="Calzetti",
            dust_prior="flat",
            Av_min=args.Av_min,
            Av_max=args.Av_max,
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

    theta_stats = {
        label: (np.mean(sx.theta[:, i]), np.std(sx.theta[:, i]))
        for i, label in enumerate(sx.labels)
    }

    # Optional training-side mock matching against real observations
    if args.mock_match != "none":
        from validate_noise_model import load_real_data, get_mock_arrays, compute_mock_match_weights

        real_data = load_real_data(args.fits_file, patch_id=args.patch_id, aperture=args.aperture)
        mock_data = get_mock_arrays(sx)
        mock_weights, match_msg = compute_mock_match_weights(
            real_data,
            mock_data,
            mode=args.mock_match,
            match_band=args.mock_match_band,
            color_band=args.mock_match_color_band,
            n_bins=args.mock_match_bins,
        )
        valid = np.isfinite(mock_weights) & (mock_weights > 0)
        if np.any(valid):
            probs = mock_weights[valid] / mock_weights[valid].sum()
            source_idx = np.where(valid)[0]
            rng = np.random.default_rng(0)
            draw_idx = rng.choice(source_idx, size=sx.n_simulation, replace=True, p=probs)
            sx.theta = sx.theta[draw_idx, :]
            sx.mag = sx.mag[draw_idx, :, :]
            sx.obs = sx.obs[draw_idx, :]
            eff_n = (mock_weights[valid].sum() ** 2) / np.sum(mock_weights[valid] ** 2)
            print(f"    {match_msg}")
            print(f"    Applied weighted resampling for training (effective N ≈ {eff_n:.0f})")
        else:
            print("    Mock matching requested, but all weights were zero. Keeping unweighted mocks.")

    # Clip pathological SFR tails
    sfr_idx = next((i for i, lab in enumerate(sx.labels) if "SFR" in lab), None)
    if sfr_idx is not None:
        before = sx.theta[:, sfr_idx].copy()
        sx.theta[:, sfr_idx] = np.clip(sx.theta[:, sfr_idx], args.sfr_floor, args.sfr_ceil)
        clipped = np.sum(before != sx.theta[:, sfr_idx])
        print(f"    Clipped SFR at [{args.sfr_floor}, {args.sfr_ceil}] for {clipped}/{len(before)} samples")

    # Temporarily drop tau and metallicity from inference targets
    if args.target_params == "no_tau_met":
        drop_idx = [
            i for i, lab in enumerate(sx.labels)
            if ("tau" in lab.lower()) or ("[m/h]" in lab.lower())
        ]
        if drop_idx:
            keep_idx = [i for i in range(sx.theta.shape[1]) if i not in drop_idx]
            dropped_labels = [sx.labels[i] for i in drop_idx]
            sx.theta = sx.theta[:, keep_idx]
            sx.labels = [sx.labels[i] for i in keep_idx]
            print(f"    Dropped target parameters: {', '.join(dropped_labels)}")

    theta_mu = None
    theta_sigma = None
    if args.theta_normalization == "zscore":
        theta_mu = np.array([theta_stats[label][0] for label in sx.labels], dtype=float)
        theta_sigma = np.array([theta_stats[label][1] for label in sx.labels], dtype=float)
        theta_sigma = np.where(theta_sigma < 1e-6, 1.0, theta_sigma)
        sx.theta = (sx.theta - theta_mu) / theta_sigma
        print("    Applied z-score normalization to training targets (stats from pre-match distribution)")

    if args.skip_train:
        print("Done: quick Euclid preparation finished (simulation + realism).")
        print("Run without --skip-train to continue to training/testing.")
        return

    print("[4/5] Training quick model...")
    n_train = len(sx.theta) if args.max_train_samples <= 0 else min(args.max_train_samples, len(sx.theta))
    min_thetas = np.min(sx.theta[:n_train], axis=0)
    max_thetas = np.max(sx.theta[:n_train], axis=0)

    sx.train(
        min_thetas=min_thetas,
        max_thetas=max_thetas,
        n_max=n_train,
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
            n_theta=sx.means_test.shape[1],
            save=True,
            name="euclid_quick_test_",
        )
        print(f"Saved diagnostic plots in {logs_dir}")

    print("Done: quick Euclid smoke test complete.")


if __name__ == "__main__":
    main()
