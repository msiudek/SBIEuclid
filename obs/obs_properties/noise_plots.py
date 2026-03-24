"""Generate comparison plots for SBIPIX noise products.

Default comparison is north_2fwhm vs north_3fwhm in obs/obs_properties.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def read_filter_labels(filter_list_path):
    with open(filter_list_path, "r", encoding="utf-8") as f:
        rel = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [os.path.splitext(os.path.basename(path))[0] for path in rel]


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
    n_filters = len(labels)
    ncols = 4
    nrows = (n_filters + ncols - 1) // ncols
    bins = np.arange(sigma_a.shape[1])

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for idx, band in enumerate(labels):
        ax = axes[idx]
        ax.plot(bins, sigma_a[idx], "-o", label=label_a)
        ax.plot(bins, sigma_b[idx], "-s", label=label_b)
        ax.set_title(band, fontsize=9)
        ax.set_xlabel("bin index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    for idx in range(n_filters, len(axes)):
        axes[idx].axis("off")

    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_percentiles_plot(labels, p_a, p_b, label_a, label_b, out_path):
    x = np.arange(len(labels))
    rows = int(np.ceil(p_a.shape[0] / 3))

    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.5 * rows), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for idx in range(p_a.shape[0]):
        ax = axes[idx]
        ax.plot(x, p_a[idx], "o-", label=label_a)
        ax.plot(x, p_b[idx], "s-", label=label_b)
        ax.set_title(f"Percentile cut {idx + 1}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("magnitude")
        ax.grid(alpha=0.3)

    for idx in range(p_a.shape[0], len(axes)):
        axes[idx].axis("off")

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate noise comparison plots")
    parser.add_argument("--base-dir", default="obs/obs_properties",
                        help="Directory containing noise product .npy files")
    parser.add_argument("--filter-list", default="filters_to_use.dat",
                        help="Filter list filename inside --base-dir")
    parser.add_argument("--prefix-a", default="north_2fwhm",
                        help="Prefix for first product set")
    parser.add_argument("--prefix-b", default="north_3fwhm",
                        help="Prefix for second product set")
    parser.add_argument("--label-a", default="2fwhm",
                        help="Legend label for first product set")
    parser.add_argument("--label-b", default="3fwhm",
                        help="Legend label for second product set")
    args = parser.parse_args()

    filter_list_path = args.filter_list
    if not os.path.isabs(filter_list_path):
        filter_list_path = os.path.join(args.base_dir, filter_list_path)

    labels = read_filter_labels(filter_list_path)
    products_a = load_products(args.base_dir, args.prefix_a)
    products_b = load_products(args.base_dir, args.prefix_b)

    save_background_plot(
        labels,
        products_a["background"],
        products_b["background"],
        args.label_a,
        args.label_b,
        os.path.join(args.base_dir, "noise_compare_background_2v3fwhm.png"),
    )

    save_sigma_grid(
        labels,
        products_a["mean_sigma"],
        products_b["mean_sigma"],
        args.label_a,
        args.label_b,
        os.path.join(args.base_dir, "noise_compare_mean_sigma_2v3fwhm.png"),
        ylabel="mean sigma [mag]",
        title="Mean sigma per bin",
    )

    save_sigma_grid(
        labels,
        products_a["std_sigma"],
        products_b["std_sigma"],
        args.label_a,
        args.label_b,
        os.path.join(args.base_dir, "noise_compare_std_sigma_2v3fwhm.png"),
        ylabel="std sigma [mag]",
        title="Std sigma per bin",
    )

    save_percentiles_plot(
        labels,
        products_a["percentiles"],
        products_b["percentiles"],
        args.label_a,
        args.label_b,
        os.path.join(args.base_dir, "noise_compare_percentiles_2v3fwhm.png"),
    )

    print("Generated plots:")
    print(f"- {os.path.join(args.base_dir, 'noise_compare_background_2v3fwhm.png')}")
    print(f"- {os.path.join(args.base_dir, 'noise_compare_mean_sigma_2v3fwhm.png')}")
    print(f"- {os.path.join(args.base_dir, 'noise_compare_std_sigma_2v3fwhm.png')}")
    print(f"- {os.path.join(args.base_dir, 'noise_compare_percentiles_2v3fwhm.png')}")


if __name__ == "__main__":
    main()
