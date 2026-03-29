from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare SBI mass/SFR posteriors against COSMOS matched catalog values"
    )
    parser.add_argument("--inference-npz", default="sbi-logs/cosmos_test_1k/cosmos_posteriors.npz")
    parser.add_argument("--matched-file", default="obs/obs_properties/matched_euclid_farmer.fits")
    parser.add_argument("--matched-id-column", default="euclid_idx")
    parser.add_argument("--catalog-id-key", default="catalog_id")
    parser.add_argument("--catalog-mass-column", default="lp_mass_med")
    parser.add_argument("--catalog-sfr-column", default="lp_sfr_med")
    parser.add_argument("--out-dir", default=None, help="Default: same directory as inference npz")
    parser.add_argument("--fig-name", default="mass_sfr_comparison.png")
    parser.add_argument("--stats-name", default="mass_sfr_comparison_stats.json")
    return parser


def _find_label_index(labels, candidates):
    labels_lower = [str(x).lower() for x in labels]
    for candidate in candidates:
        c = candidate.lower()
        for idx, label in enumerate(labels_lower):
            if c in label:
                return idx
    return None


def _build_value_map(ids, values):
    ids = np.asarray(ids)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    ids = ids[ok]
    values = values[ok]

    try:
        ids_i64 = ids.astype(np.int64)
        order = np.argsort(ids_i64)
        ids_i64 = ids_i64[order]
        values = values[order]
        first = np.unique(ids_i64, return_index=True)[1]
        return ids_i64[first], values[first], "int"
    except Exception:
        ids_s = np.array([str(x).strip() for x in ids], dtype=object)
        order = np.argsort(ids_s)
        ids_s = ids_s[order]
        values = values[order]
        first = np.unique(ids_s, return_index=True)[1]
        return ids_s[first], values[first], "str"


def _lookup(ids_query, ids_ref_unique, vals_ref_unique, mode):
    out = np.full(len(ids_query), np.nan, dtype=float)

    if mode == "int":
        q = np.asarray(ids_query).astype(np.int64)
        idx = np.searchsorted(ids_ref_unique, q)
        hit = idx < len(ids_ref_unique)
        hit[hit] &= ids_ref_unique[idx[hit]] == q[hit]
        out[hit] = vals_ref_unique[idx[hit]]
        return out

    q = np.array([str(x).strip() for x in ids_query], dtype=object)
    idx = np.searchsorted(ids_ref_unique, q)
    hit = idx < len(ids_ref_unique)
    hit[hit] &= ids_ref_unique[idx[hit]] == q[hit]
    out[hit] = vals_ref_unique[idx[hit]]
    return out


def _compute_stats(pred, truth):
    diff = pred - truth
    bias = float(np.nanmedian(diff))
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    mad_sigma = float(1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff))))
    corr = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 else np.nan
    return {
        "n": int(len(diff)),
        "bias_median": bias,
        "rmse": rmse,
        "sigma_mad": mad_sigma,
        "pearson_r": corr,
    }


def _plot_scatter_with_identity(ax, truth, pred, title, xlabel, ylabel):
    ax.scatter(truth, pred, s=10, alpha=0.35)
    vmin = float(np.nanpercentile(np.concatenate([truth, pred]), 1))
    vmax = float(np.nanpercentile(np.concatenate([truth, pred]), 99))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.2)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _plot_residual(ax, truth, pred, title, xlabel):
    resid = pred - truth
    ax.scatter(truth, resid, s=10, alpha=0.35)
    ax.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SBI - COSMOS")
    ax.grid(alpha=0.25)


def main():
    args = build_parser().parse_args()

    inference_path = Path(args.inference_npz)
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference file not found: {inference_path}")

    out_dir = Path(args.out_dir) if args.out_dir else inference_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(inference_path, allow_pickle=True)
    med = np.asarray(data["posterior_median"], dtype=float)
    labels = [str(x) for x in np.asarray(data["labels"], dtype=object)]
    if args.catalog_id_key not in data.files:
        raise KeyError(f"Missing key in npz: {args.catalog_id_key}")
    catalog_ids = np.asarray(data[args.catalog_id_key])

    mass_idx = _find_label_index(labels, ["m}_{*", "log($\\rm{m}_{*}", "stellar", "mass"])
    sfr_idx = _find_label_index(labels, ["sfr", "log(sfr"])
    if mass_idx is None or sfr_idx is None:
        raise ValueError(f"Could not auto-detect mass/sfr labels from: {labels}")

    mass_pred = med[:, mass_idx]
    sfr_pred = med[:, sfr_idx]

    matched = Table.read(args.matched_file)
    if args.matched_id_column not in matched.colnames:
        raise ValueError(f"Missing matched ID column: {args.matched_id_column}")
    for c in [args.catalog_mass_column, args.catalog_sfr_column]:
        if c not in matched.colnames:
            raise ValueError(f"Missing catalog column in matched file: {c}")

    ids_unique_mass, vals_unique_mass, mode_mass = _build_value_map(
        matched[args.matched_id_column], matched[args.catalog_mass_column]
    )
    ids_unique_sfr, vals_unique_sfr, mode_sfr = _build_value_map(
        matched[args.matched_id_column], matched[args.catalog_sfr_column]
    )

    mass_true = _lookup(catalog_ids, ids_unique_mass, vals_unique_mass, mode_mass)
    sfr_true = _lookup(catalog_ids, ids_unique_sfr, vals_unique_sfr, mode_sfr)

    ok_mass = np.isfinite(mass_pred) & np.isfinite(mass_true)
    ok_sfr = np.isfinite(sfr_pred) & np.isfinite(sfr_true)

    if np.sum(ok_mass) < 5:
        raise ValueError("Too few matched finite mass values to plot")
    if np.sum(ok_sfr) < 5:
        raise ValueError("Too few matched finite SFR values to plot")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    _plot_scatter_with_identity(
        axes[0, 0],
        mass_true[ok_mass],
        mass_pred[ok_mass],
        "Stellar Mass: SBI vs COSMOS",
        f"COSMOS {args.catalog_mass_column}",
        "SBI posterior median",
    )
    _plot_residual(
        axes[0, 1],
        mass_true[ok_mass],
        mass_pred[ok_mass],
        "Stellar Mass Residual",
        f"COSMOS {args.catalog_mass_column}",
    )

    _plot_scatter_with_identity(
        axes[1, 0],
        sfr_true[ok_sfr],
        sfr_pred[ok_sfr],
        "SFR: SBI vs COSMOS",
        f"COSMOS {args.catalog_sfr_column}",
        "SBI posterior median",
    )
    _plot_residual(
        axes[1, 1],
        sfr_true[ok_sfr],
        sfr_pred[ok_sfr],
        "SFR Residual",
        f"COSMOS {args.catalog_sfr_column}",
    )

    plt.tight_layout()
    fig_path = out_dir / args.fig_name
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)

    mass_stats = _compute_stats(mass_pred[ok_mass], mass_true[ok_mass])
    sfr_stats = _compute_stats(sfr_pred[ok_sfr], sfr_true[ok_sfr])

    stats = {
        "inference_npz": str(inference_path),
        "matched_file": str(Path(args.matched_file)),
        "mass_label": labels[mass_idx],
        "sfr_label": labels[sfr_idx],
        "n_objects_in_npz": int(len(catalog_ids)),
        "n_mass_compared": int(np.sum(ok_mass)),
        "n_sfr_compared": int(np.sum(ok_sfr)),
        "mass": mass_stats,
        "sfr": sfr_stats,
    }

    stats_path = out_dir / args.stats_name
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved figure: {fig_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Mass stats: {mass_stats}")
    print(f"SFR stats: {sfr_stats}")


if __name__ == "__main__":
    main()
