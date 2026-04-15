"""Generate comparison plots for noise products.

Compares a reference photometry type against one or more other types.
By default: 2fwhm vs [3fwhm, templfit, sersic] when files exist.
Plots are saved in ./plots (created automatically).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def read_filter_labels(filter_list_path):
    """Read short filter names (col 2) from 3-column filters_to_use.dat."""
    labels = []
    with open(filter_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            labels.append(parts[1] if len(parts) >= 2 else parts[0])
    return labels


def load_products(base_dir, prefix):
    return {
        "mean_sigma": np.load(os.path.join(base_dir, f"mean_sigma_{prefix}.npy")),
        "std_sigma": np.load(os.path.join(base_dir, f"std_sigma_{prefix}.npy")),
        "background": np.load(os.path.join(base_dir, f"background_noise_{prefix}.npy")),
        "percentiles": np.load(os.path.join(base_dir, f"percentiles_{prefix}.npy")),
    }


def products_exist(base_dir, prefix):
    needed = [
        f"mean_sigma_{prefix}.npy",
        f"std_sigma_{prefix}.npy",
        f"background_noise_{prefix}.npy",
        f"percentiles_{prefix}.npy",
    ]
    return all(os.path.exists(os.path.join(base_dir, n)) for n in needed)


def build_prefix(hemisphere, phot_type):
    return f"{hemisphere}_{phot_type}"


def save_background_plot(labels, bg_a, bg_b, label_a, label_b, out_path):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(x, bg_a, "o-", label=label_a)
    ax.plot(x, bg_b, "s-", label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("background_noise [uJy]")
    ax.set_title("Background noise comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_sigma_grid(labels, sigma_a, sigma_b, label_a, label_b, out_path, ylabel, title):
    n_filters = 10
    bins = np.arange(sigma_a.shape[1])

    fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=True)
    axes = axes.ravel()

    for idx in range(n_filters):
        band = labels[idx]
        ax = axes[idx]
        ax.plot(bins, sigma_a[idx], "-o", label=label_a)
        ax.plot(bins, sigma_b[idx], "-s", label=label_b)
        ax.set_title(band, fontsize=9)
        ax.set_xlabel("bin index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_percentiles_plot(labels, p_a, p_b, label_a, label_b, out_path):
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=False)
    axes = axes.ravel()

    for idx in range(6):
        ax = axes[idx]
        ax.plot(x, p_a[idx], "o-", label=label_a)
        ax.plot(x, p_b[idx], "s-", label=label_b)
        ax.set_title(f"Percentile cut {idx + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("magnitude")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot noise-product comparisons")
    p.add_argument("--hemisphere", type=str, default="north", help="Noise-file hemisphere prefix")
    p.add_argument("--ref-type", type=str, default="2fwhm", help="Reference photometry type")
    p.add_argument(
        "--compare-types",
        nargs="+",
        default=["3fwhm", "templfit", "sersic"],
        help="Photometry types to compare against ref-type",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    prefix_a = build_prefix(args.hemisphere, args.ref_type)
    label_a = prefix_a

    filter_list_path = os.path.join(base_dir, "filters_to_use.dat")
    labels = read_filter_labels(filter_list_path)
    if not products_exist(base_dir, prefix_a):
        raise FileNotFoundError(f"Missing noise products for reference type: {prefix_a}")
    products_a = load_products(base_dir, prefix_a)

    generated = []
    for phot_type in args.compare_types:
        prefix_b = build_prefix(args.hemisphere, phot_type)
        if not products_exist(base_dir, prefix_b):
            print(f"Skipping {prefix_b}: missing one or more required .npy files")
            continue

        label_b = prefix_b
        products_b = load_products(base_dir, prefix_b)
        tag = f"{args.ref_type}_vs_{phot_type}"

        bg_out = os.path.join(plots_dir, f"noise_compare_background_{tag}.png")
        save_background_plot(
            labels,
            products_a["background"],
            products_b["background"],
            label_a,
            label_b,
            bg_out,
        )
        generated.append(bg_out)

        mean_out = os.path.join(plots_dir, f"noise_compare_mean_sigma_{tag}.png")
        save_sigma_grid(
            labels,
            products_a["mean_sigma"],
            products_b["mean_sigma"],
            label_a,
            label_b,
            mean_out,
            ylabel="mean sigma [mag]",
            title="Mean sigma per bin",
        )
        generated.append(mean_out)

        std_out = os.path.join(plots_dir, f"noise_compare_std_sigma_{tag}.png")
        save_sigma_grid(
            labels,
            products_a["std_sigma"],
            products_b["std_sigma"],
            label_a,
            label_b,
            std_out,
            ylabel="std sigma [mag]",
            title="Std sigma per bin",
        )
        generated.append(std_out)

        pct_out = os.path.join(plots_dir, f"noise_compare_percentiles_{tag}.png")
        save_percentiles_plot(
            labels,
            products_a["percentiles"],
            products_b["percentiles"],
            label_a,
            label_b,
            pct_out,
        )
        generated.append(pct_out)

    if generated:
        print("Generated plots:")
        for p in generated:
            print(f"- {p}")
    else:
        print("No plots generated (all requested compare types were missing).")


if __name__ == "__main__":
    main()
