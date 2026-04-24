"""
Mock-as-data selection sweep for SBIPIX mass/SFR inference.

Purpose
-------
Run the same inference path as COSMOS-Web (noisy flux-like observations,
_get_posterior_obs, redshift conditioning), but compare against true atlas
parameters. Sweep multiple selection cuts (snr_min, n_bands_min) to test if
mass offset is selection-driven.

Example
-------
python examples/mock_as_data_selection_sweep.py \
  --model-name model_euclid_v5.0_20k_flux.pkl \
  --atlas-name atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas \
  --n-sim 20000 \
  --phot-type templfit \
  --observation-space flux \
  --snr-grid 2 3 5 \
  --n-bands-grid 5 7 9 \
  --n-gal 2000 \
  --n-samples 80 \
  --device cuda \
  --outdir sbi-logs/mock_as_data_sweep_v5.0
"""

import argparse
import io
import pickle
import re
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Mock-as-data sweep for selection-driven bias tests")
    p.add_argument(
        "--model-name",
        type=str,
        default="model_euclid_v5.0_20k_flux.pkl",
        help="Model filename in library/",
    )
    p.add_argument(
        "--atlas-name",
        type=str,
        default="atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas",
        help="Atlas filename in library/",
    )
    p.add_argument("--n-sim", type=int, default=20000, help="Number of atlas simulations")
    p.add_argument(
        "--phot-type",
        type=str,
        default="templfit",
        choices=["templfit", "2fwhm", "3fwhm"],
        help="Photometry type",
    )
    p.add_argument(
        "--observation-space",
        type=str,
        default="flux",
        choices=["mag", "flux"],
        help="Observation feature space",
    )
    p.add_argument(
        "--sigma-sampler",
        type=str,
        default="mag_lognormal",
        choices=["empirical", "truncnorm", "mag_lognormal"],
        help="Noise sigma sampler",
    )
    p.add_argument(
        "--detection-model",
        type=str,
        default="hard",
        choices=["hard", "probabilistic"],
        help="Detection model",
    )
    p.add_argument(
        "--snr-grid",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 5.0],
        help="List of SNR cuts to test",
    )
    p.add_argument(
        "--n-bands-grid",
        type=int,
        nargs="+",
        default=[5, 7, 9],
        help="List of minimum detected-band cuts to test",
    )
    p.add_argument(
        "--n-gal",
        type=int,
        default=2000,
        help="Maximum galaxies per grid cell",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=80,
        help="Posterior samples per galaxy",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: cuda or cpu",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="sbi-logs/mock_as_data_sweep",
        help="Output directory",
    )
    return p.parse_args()


def normalize_atlas_name(atlas_name, n_sim, n_param=2):
    value = str(atlas_name).strip()
    if value.endswith(".dbatlas"):
        value = value[: -len(".dbatlas")]

    suffix = f"_{int(n_sim)}_Nparam_{int(n_param)}"
    if value.endswith(suffix):
        value = value[: -len(suffix)]

    value = re.sub(r"_\d+_Nparam_\d+$", "", value)
    return value


class DeviceUnpickler(pickle.Unpickler):
    """Unpickle torch storages onto a selected device."""

    def __init__(self, file_obj, map_location):
        super().__init__(file_obj)
        self._map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=self._map_location)
        return super().find_class(module, name)


def load_pickle_device_safe(path, device):
    with open(path, "rb") as f:
        return DeviceUnpickler(f, map_location=device).load()


def probe_context_support(qphi, context_dim, device):
    try:
        qphi.sample(
            (1,),
            x=torch.zeros((1, context_dim), dtype=torch.float32, device=device),
            show_progress_bars=False,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)


def nmad(x):
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    root = Path(__file__).resolve().parents[1]
    obs_dir = root / "obs" / "obs_properties"
    lib_dir = root / "library"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from sbipix import sbipix

    noise_prefix = f"north_{args.phot_type}"

    sx = sbipix()
    sx.configure_filters(
        filter_list="filters_to_use.dat",
        filter_path=str(obs_dir),
        mean_sigma_file=f"mean_sigma_{noise_prefix}.npy",
        std_sigma_file=f"std_sigma_{noise_prefix}.npy",
        percentiles_file=f"percentiles_{noise_prefix}.npy",
        limits_file=f"background_noise_{noise_prefix}.npy",
        lam_eff_file=f"lam_eff_{noise_prefix}.npy",
    )

    sx.atlas_path = str(lib_dir) + "/"
    sx.model_path = str(lib_dir) + "/"
    sx.atlas_name = normalize_atlas_name(args.atlas_name, args.n_sim, 2)
    sx.model_name = args.model_name
    sx.n_simulation = args.n_sim

    sx.parametric = True
    sx.both_masses = True
    sx.infer_z = False
    sx.include_limit = True
    sx.include_sigma = True
    sx.condition_sigma = True

    sx.configure_noise_model(
        sigma_sampler=args.sigma_sampler,
        detection_model=args.detection_model,
        observation_space=args.observation_space,
    )

    print("Loading simulation and creating noisy mock observations...")
    sx.load_simulation()
    sx.load_obs_features()
    sx.add_noise_nan_limit_all()

    ok = np.isfinite(np.sum(sx.theta, axis=1))
    phys_ok = (
        (sx.theta[:, 0] > 4.0)
        & (sx.theta[:, 0] < 13.0)
        & (sx.theta[:, 2] > -4.0)
        & (sx.theta[:, 2] < 3.0)
    )
    base_mask = ok & phys_ok

    flux_obs = np.asarray(sx.mag[:, :, 0], dtype=float)
    sigma_obs = np.asarray(sx.mag[:, :, 1], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(flux_obs / np.where(sigma_obs > 0, sigma_obs, np.nan))

    obs_raw = np.reshape(sx.mag, (len(sx.mag), sx.mag.shape[1] * sx.mag.shape[2]))

    model_file = str(lib_dir / args.model_name)
    anpe_file = str(lib_dir / f"anpe_{args.model_name}")

    qphi = None
    probe_errors = []

    if Path(anpe_file).exists():
        try:
            anpe = load_pickle_device_safe(anpe_file, args.device)
            try:
                qphi = anpe.build_posterior(sample_with="rejection")
            except TypeError:
                qphi = anpe.build_posterior()
            print(f"Loaded posterior from {anpe_file}")
        except Exception as exc:
            probe_errors.append(f"anpe load/build failed ({exc})")
            qphi = None
    else:
        probe_errors.append("anpe file missing")

    if qphi is None:
        qphi = load_pickle_device_safe(model_file, args.device)
        print(f"Loaded posterior from {model_file}")

    context_dim = obs_raw.shape[1] + 1
    ok_ctx, err_ctx = probe_context_support(qphi, context_dim, args.device)
    if not ok_ctx:
        raise RuntimeError(
            "Posterior does not support obs+z context for this run. "
            f"context_dim={context_dim}, error={err_ctx}, prior_probe={probe_errors}"
        )

    rng = np.random.default_rng(args.seed)


    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    rows = []


    print("Running selection sweep...")
    plot_grid = {}
    for snr_min in args.snr_grid:
        n_bands = np.sum((snr >= snr_min) & np.isfinite(snr), axis=1)

        for n_bands_min in args.n_bands_grid:
            sel = base_mask & (n_bands >= n_bands_min)
            idx_all = np.where(sel)[0]

            if len(idx_all) == 0:
                print(f"snr_min={snr_min:.1f}, n_bands_min={n_bands_min}: no galaxies")
                rows.append(
                    {
                        "snr_min": float(snr_min),
                        "n_bands_min": int(n_bands_min),
                        "n_candidates": 0,
                        "n_used": 0,
                        "mass_slope": np.nan,
                        "mass_intercept": np.nan,
                        "mass_median_delta": np.nan,
                        "mass_nmad": np.nan,
                        "sfr_slope": np.nan,
                        "sfr_intercept": np.nan,
                        "sfr_median_delta": np.nan,
                        "sfr_nmad": np.nan,
                        "corr_deltaM_z": np.nan,
                        "mass_r2": np.nan,
                        "sfr_r2": np.nan,
                    }
                )
                continue

            if len(idx_all) > args.n_gal:
                idx = np.sort(rng.choice(idx_all, size=args.n_gal, replace=False))
            else:
                idx = np.sort(idx_all)

            obs_sel = obs_raw[idx]
            z_sel = sx.theta[idx, 7]
            true_mass = sx.theta[idx, 0]
            true_sfr = sx.theta[idx, 2]

            post = sx._get_posterior_obs(
                obs_sel,
                qphi,
                n_samples=args.n_samples,
                bar=True,
                input_z=z_sel,
                device=args.device,
            )

            pred_mass = np.nanmedian(post[:, :, 0], axis=1)
            pred_sfr = np.nanmedian(post[:, :, 1], axis=1)

            d_mass = pred_mass - true_mass
            d_sfr = pred_sfr - true_sfr

            if len(pred_mass) >= 2:
                mass_slope, mass_intercept = np.polyfit(true_mass, pred_mass, 1)
                sfr_slope, sfr_intercept = np.polyfit(true_sfr, pred_sfr, 1)
                corr_deltaM_z = np.corrcoef(d_mass, z_sel)[0, 1]
                mass_r2 = r2_score(true_mass, pred_mass)
                sfr_r2 = r2_score(true_sfr, pred_sfr)
            else:
                mass_slope, mass_intercept = np.nan, np.nan
                sfr_slope, sfr_intercept = np.nan, np.nan
                corr_deltaM_z = np.nan
                mass_r2 = np.nan
                sfr_r2 = np.nan

            row = {
                "snr_min": float(snr_min),
                "n_bands_min": int(n_bands_min),
                "n_candidates": int(len(idx_all)),
                "n_used": int(len(idx)),
                "mass_slope": float(mass_slope),
                "mass_intercept": float(mass_intercept),
                "mass_median_delta": float(np.nanmedian(d_mass)),
                "mass_nmad": float(nmad(d_mass)),
                "sfr_slope": float(sfr_slope),
                "sfr_intercept": float(sfr_intercept),
                "sfr_median_delta": float(np.nanmedian(d_sfr)),
                "sfr_nmad": float(nmad(d_sfr)),
                "corr_deltaM_z": float(corr_deltaM_z),
                "mass_r2": float(mass_r2),
                "sfr_r2": float(sfr_r2),
            }
            rows.append(row)

            # Plot inferred vs true for logM
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            lim_mass = [np.nanmin([true_mass, pred_mass]), np.nanmax([true_mass, pred_mass])]
            lim_sfr = [np.nanmin([true_sfr, pred_sfr]), np.nanmax([true_sfr, pred_sfr])]

            axes[0].scatter(true_mass, pred_mass, s=8, alpha=0.5, color="tab:blue")
            axes[0].plot(lim_mass, lim_mass, "k--", lw=1)
            axes[0].set_xlabel("True logM")
            axes[0].set_ylabel("Inferred logM")
            axes[0].set_title(f"logM R2={mass_r2:.3f}")
            axes[0].set_xlim(lim_mass)
            axes[0].set_ylim(lim_mass)

            axes[1].scatter(true_sfr, pred_sfr, s=8, alpha=0.5, color="tab:orange")
            axes[1].plot(lim_sfr, lim_sfr, "k--", lw=1)
            axes[1].set_xlabel("True logSFR")
            axes[1].set_ylabel("Inferred logSFR")
            axes[1].set_title(f"logSFR R2={sfr_r2:.3f}")
            axes[1].set_xlim(lim_sfr)
            axes[1].set_ylim(lim_sfr)

            plt.suptitle(f"SNR>={snr_min}, Nbands>={n_bands_min} (n={len(idx)})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_name = outdir / f"scatter_logM_logSFR_snr{snr_min}_nbands{n_bands_min}.png"
            plt.savefig(plot_name)
            plt.close(fig)
            plot_grid[(snr_min, n_bands_min)] = plot_name

            print(
                f"snr>={snr_min:.1f}, n_bands>={n_bands_min}: "
                f"n={row['n_used']}, dM_med={row['mass_median_delta']:+.3f}, "
                f"dM_NMAD={row['mass_nmad']:.3f}, corr(dM,z)={row['corr_deltaM_z']:+.3f}, "
                f"R2_M={mass_r2:.3f}, R2_SFR={sfr_r2:.3f}"
            )


    rows_sorted = sorted(rows, key=lambda r: (r["snr_min"], r["n_bands_min"]))


    csv_file = outdir / "selection_sweep_summary.csv"
    headers = [
        "snr_min",
        "n_bands_min",
        "n_candidates",
        "n_used",
        "mass_slope",
        "mass_intercept",
        "mass_median_delta",
        "mass_nmad",
        "sfr_slope",
        "sfr_intercept",
        "sfr_median_delta",
        "sfr_nmad",
        "corr_deltaM_z",
        "mass_r2",
        "sfr_r2",
    ]


    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows_sorted:
            vals = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    vals.append(f"{v:.10g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    # Optionally, plot a summary grid of R2 for logM and logSFR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        snr_vals = sorted(set(r["snr_min"] for r in rows_sorted))
        nbands_vals = sorted(set(r["n_bands_min"] for r in rows_sorted))
        r2_mass = np.full((len(snr_vals), len(nbands_vals)), np.nan)
        r2_sfr = np.full((len(snr_vals), len(nbands_vals)), np.nan)
        for r in rows_sorted:
            i = snr_vals.index(r["snr_min"])
            j = nbands_vals.index(r["n_bands_min"])
            r2_mass[i, j] = r["mass_r2"]
            r2_sfr[i, j] = r["sfr_r2"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(r2_mass, origin="lower", aspect="auto", vmin=0, vmax=1)
        axes[0].set_xticks(np.arange(len(nbands_vals)))
        axes[0].set_xticklabels(nbands_vals)
        axes[0].set_yticks(np.arange(len(snr_vals)))
        axes[0].set_yticklabels(snr_vals)
        axes[0].set_xlabel("n_bands_min")
        axes[0].set_ylabel("snr_min")
        axes[0].set_title("logM R2")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(r2_sfr, origin="lower", aspect="auto", vmin=0, vmax=1)
        axes[1].set_xticks(np.arange(len(nbands_vals)))
        axes[1].set_xticklabels(nbands_vals)
        axes[1].set_yticks(np.arange(len(snr_vals)))
        axes[1].set_yticklabels(snr_vals)
        axes[1].set_xlabel("n_bands_min")
        axes[1].set_ylabel("snr_min")
        axes[1].set_title("logSFR R2")
        fig.colorbar(im1, ax=axes[1])

        plt.suptitle("R2 summary for logM and logSFR")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(outdir / "r2_summary_grid.png")
        plt.close(fig)
    except Exception as e:
        print(f"Could not plot R2 summary grid: {e}")


    npz_file = outdir / "selection_sweep_summary.npz"
    np.savez(npz_file, rows=np.array(rows_sorted, dtype=object))

    print("\nSaved outputs:")
    print(f"  {csv_file}")
    print(f"  {npz_file}")
    print(f"  Scatter plots for each grid cell in {outdir}")
    print(f"  R2 summary grid: {outdir / 'r2_summary_grid.png'}")


if __name__ == "__main__":
    main()
