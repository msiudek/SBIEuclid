"""Generate comparison plots for noise products.

Simple version: compares 2fwhm vs 3fwhm products in this directory.
Plots are saved in ./plots (created automatically).
"""

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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    prefix_a = "north_2fwhm"
    prefix_b = "north_3fwhm"
    label_a = "north_2fwhm"
    label_b = "north_3fwhm"

    filter_list_path = os.path.join(base_dir, "filters_to_use.dat")
    labels = read_filter_labels(filter_list_path)
    products_a = load_products(base_dir, prefix_a)
    products_b = load_products(base_dir, prefix_b)

    save_background_plot(
        labels,
        products_a["background"],
        products_b["background"],
        label_a,
        label_b,
        os.path.join(plots_dir, "noise_compare_background_2v3fwhm.png"),
    )

    save_sigma_grid(
        labels,
        products_a["mean_sigma"],
        products_b["mean_sigma"],
        label_a,
        label_b,
        os.path.join(plots_dir, "noise_compare_mean_sigma_2v3fwhm.png"),
        ylabel="mean sigma [mag]",
        title="Mean sigma per bin",
    )

    save_sigma_grid(
        labels,
        products_a["std_sigma"],
        products_b["std_sigma"],
        label_a,
        label_b,
        os.path.join(plots_dir, "noise_compare_std_sigma_2v3fwhm.png"),
        ylabel="std sigma [mag]",
        title="Std sigma per bin",
    )

    save_percentiles_plot(
        labels,
        products_a["percentiles"],
        products_b["percentiles"],
        label_a,
        label_b,
        os.path.join(plots_dir, "noise_compare_percentiles_2v3fwhm.png"),
    )

    print("Generated plots:")
    print(f"- {os.path.join(plots_dir, 'noise_compare_background_2v3fwhm.png')}")
    print(f"- {os.path.join(plots_dir, 'noise_compare_mean_sigma_2v3fwhm.png')}")
    print(f"- {os.path.join(plots_dir, 'noise_compare_std_sigma_2v3fwhm.png')}")
    print(f"- {os.path.join(plots_dir, 'noise_compare_percentiles_2v3fwhm.png')}")


if __name__ == "__main__":
    main()
