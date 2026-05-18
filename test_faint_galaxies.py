"""
Test whether sigma mismatch affects inference on faint galaxies.

Runs inference on a subset of faint galaxies (NISP-Y mag 27-28) and checks:
1. If results are sensible (posteriors not degenerate)
2. Mass bias patterns compared to bright galaxies
3. Whether updated background_noise calibration is needed

Usage:
    python test_faint_galaxies.py --model-name model_euclid_v6.3_20k.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import json
from astropy.table import Table

def run_inference_subset(mag_min, mag_max, mag_band="NISP-Y", suffix=""):
    """Run inference on galaxies in magnitude range."""
    outdir = f"sbi-logs/test_faint_gal_{mag_min:.1f}_{mag_max:.1f}_{suffix}"
    cmd = [
        "python", "examples/inference_cosmosweb.py",
        "--n-gal", "100",
        "--n-samples", "200",
        "--mag-band", mag_band,
        "--mag-min", str(mag_min),
        "--mag-max", str(mag_max),
        "--outdir", outdir,
        "--device", "cuda",
        "--n-bands-min", "7",
        "--snr-min", "3",
        "--phot-type", "templfit",
        "--sample-with", "rejection",
        "--observation-space", "flux",
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed for {mag_min}-{mag_max}")
    return Path(outdir)

def analyze_results(result_dir):
    """Analyze inference results to check for sanity."""
    result_file = result_dir / "inference_results.npz"
    if not result_file.exists():
        raise FileNotFoundError(f"Results not found: {result_file}")
    
    data = np.load(result_file)
    logM_sbi = data["logM_sbi"]
    logM_ref = data["logM_cosmosweb"]
    z = data["z"]
    posteriors = data["posteriors"]  # shape: (n_gal, n_samples, n_theta)
    
    # Check 1: Are posteriors degenerate?
    posterior_widths = np.nanstd(posteriors[:, :, 0], axis=1)  # std of logM samples
    degenerate = posterior_widths < 0.05  # very narrow posteriors
    
    # Check 2: Compute mass bias
    delta_logm = logM_sbi - logM_ref
    bias_median = np.nanmedian(delta_logm)
    bias_std = np.nanstd(delta_logm)
    bias_correlation_z = np.corrcoef(delta_logm[np.isfinite(delta_logm)], 
                                     z[np.isfinite(delta_logm)])[0, 1]
    
    # Check 3: Number of valid inferences
    n_valid = np.sum(np.isfinite(logM_sbi))
    
    stats = {
        "n_gal": len(logM_sbi),
        "n_valid": n_valid,
        "z_mean": float(np.nanmean(z)),
        "z_range": (float(np.nanmin(z)), float(np.nanmax(z))),
        "posterior_width_median": float(np.nanmedian(posterior_widths)),
        "posterior_width_std": float(np.nanstd(posterior_widths)),
        "n_degenerate": int(np.sum(degenerate)),
        "bias_median": float(bias_median),
        "bias_std": float(bias_std),
        "bias_correlation_with_z": float(bias_correlation_z),
        "bias_percentiles": {
            "p16": float(np.nanpercentile(delta_logm, 16)),
            "p50": float(np.nanpercentile(delta_logm, 50)),
            "p84": float(np.nanpercentile(delta_logm, 84)),
        },
    }
    
    return stats

def plot_comparison(results_dict):
    """Plot bias and posterior properties for different magnitude ranges."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    mag_ranges = sorted(results_dict.keys())
    labels = [f"{mag:.1f}-{mag+1:.1f}" for mag, _ in mag_ranges]
    
    biases = [results_dict[mag_range]["bias_median"] for mag_range in mag_ranges]
    bias_stds = [results_dict[mag_range]["bias_std"] for mag_range in mag_ranges]
    posterior_widths = [results_dict[mag_range]["posterior_width_median"] for mag_range in mag_ranges]
    z_means = [results_dict[mag_range]["z_mean"] for mag_range in mag_ranges]
    
    # Bias vs magnitude
    axes[0, 0].errorbar(range(len(mag_ranges)), biases, yerr=bias_stds, marker='o', linestyle='-')
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xticks(range(len(mag_ranges)))
    axes[0, 0].set_xticklabels(labels, rotation=45)
    axes[0, 0].set_ylabel("Median ΔlogM* (dex)")
    axes[0, 0].set_title("Mass Bias vs Magnitude Range")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Posterior width vs magnitude
    axes[0, 1].plot(range(len(mag_ranges)), posterior_widths, marker='o', linestyle='-', color='orange')
    axes[0, 1].set_xticks(range(len(mag_ranges)))
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].set_ylabel("Median Posterior σ(logM*) (dex)")
    axes[0, 1].set_title("Posterior Width vs Magnitude Range")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean redshift vs magnitude
    axes[1, 0].plot(range(len(mag_ranges)), z_means, marker='s', linestyle='-', color='green')
    axes[1, 0].set_xticks(range(len(mag_ranges)))
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].set_ylabel("Mean Redshift")
    axes[1, 0].set_title("Mean Redshift vs Magnitude Range")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = "Faint Galaxy Inference Test Summary:\n\n"
    for i, mag_range in enumerate(mag_ranges):
        stats = results_dict[mag_range]
        summary_text += (
            f"Mag {labels[i]}: "
            f"bias={stats['bias_median']:+.3f}±{stats['bias_std']:.3f} dex, "
            f"width={stats['posterior_width_median']:.3f} dex, "
            f"n_valid={stats['n_valid']}/{stats['n_gal']}\n"
        )
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("sbi-logs/test_faint_gal_summary.png", dpi=150, bbox_inches='tight')
    print("Plot saved to sbi-logs/test_faint_gal_summary.png")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Test sigma mismatch impact on faint galaxies")
    p.add_argument("--model-name", type=str, default="model_euclid_v9.0.pkl",
                   help="Model file to use for inference")
    p.add_argument("--skip-bright", action="store_true",
                   help="Skip bright galaxy test (mag 22-24)")
    args = p.parse_args()
    
    print("=" * 70)
    print("Testing sigma mismatch impact on faint galaxies")
    print("=" * 70)
    
    # Define magnitude ranges to test
    mag_ranges_to_test = [
        (22, 24, "bright"),
        (24, 26, "intermediate"),
        (26, 28, "faint"),
        (27, 28, "very_faint"),
    ]
    
    results = {}
    
    for mag_min, mag_max, label in mag_ranges_to_test:
        if label == "bright" and args.skip_bright:
            print(f"\nSkipping {label} galaxies (mag {mag_min}-{mag_max})")
            continue
            
        print(f"\n{'='*70}")
        print(f"Testing {label} galaxies (NISP-Y mag {mag_min}-{mag_max})")
        print(f"{'='*70}")
        
        try:
            outdir = run_inference_subset(mag_min, mag_max, suffix=label)
            stats = analyze_results(outdir)
            results[(mag_min, mag_max)] = stats
            
            print(f"\nResults for {label} (mag {mag_min}-{mag_max}):")
            print(f"  Galaxies: {stats['n_valid']}/{stats['n_gal']} valid inferences")
            print(f"  Redshift range: {stats['z_range'][0]:.2f} - {stats['z_range'][1]:.2f} (mean={stats['z_mean']:.2f})")
            print(f"  Mass bias: {stats['bias_median']:+.3f} ± {stats['bias_std']:.3f} dex")
            print(f"  Bias correlation with z: {stats['bias_correlation_with_z']:.3f}")
            print(f"  Posterior width: {stats['posterior_width_median']:.3f} ± {stats['posterior_width_std']:.3f} dex")
            print(f"  Degenerate posteriors (width < 0.05): {stats['n_degenerate']}")
            
            if stats['n_degenerate'] > 0:
                print(f"  ⚠️  {stats['n_degenerate']} degenerate posteriors detected!")
            
            if stats['bias_median'] > 0.5:
                print(f"  ⚠️  Large positive bias ({stats['bias_median']:+.3f} dex) detected!")
                print(f"     This suggests sigma mismatch or detection model issues")
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison plot
    if len(results) > 1:
        print("\n" + "="*70)
        print("Creating comparison plot...")
        plot_comparison(results)
    
    # Save results JSON
    json_path = Path("sbi-logs/test_faint_gal_results.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    for (mag_min, mag_max), stats in sorted(results.items()):
        print(f"Mag {mag_min:.0f}-{mag_max:.0f}: bias={stats['bias_median']:+.3f} dex, "
              f"width={stats['posterior_width_median']:.3f} dex")
    
    # Decision: do we need to update background_noise?
    very_faint_bias = results.get((27, 28), {}).get("bias_median", np.nan)
    if not np.isnan(very_faint_bias) and np.abs(very_faint_bias) > 0.3:
        print("\n" + "⚠️  " + "="*66)
        print("RECOMMENDATION: Large bias detected in very faint galaxies!")
        print("This suggests the background_noise calibration needs updating.")
        print("Consider re-running: python examples/validate_noise_model.py")
        print("and checking if templfit limits match true detection limits.")
        print("="*70)

if __name__ == "__main__":
    main()
