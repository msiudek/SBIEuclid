"""Per-galaxy forensic: how do retrained flows differ from the June-22 v3 model?

Compares logM point estimates galaxy-by-galaxy between inference runs on the
identical Khostovan input catalog. If the difference is a uniform zero-point
(flat in z and mass, small scatter), the flows learned the same mapping shape
up to a constant mass normalization; if it is structured, they disagree in a
specific regime.

Usage:
    python compare_v3_vs_retrain.py \
        --ref sbi-logs/inference_khostovan_v3_total \
        --run sbi-logs/inference_khostovan_JWnoise sbi-logs/inference_khostovan_sw_big_lr1e4 \
        --out sbi-logs/fig_v3_vs_retrain.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(path):
    d = np.load(os.path.join(path, "inference_results.npz"), allow_pickle=True)
    keys = list(d.keys())
    out = {k: d[k] for k in keys}
    return out, keys


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference run dir (v3)")
    ap.add_argument("--run", nargs="+", required=True, help="retrain run dirs")
    ap.add_argument("--out", default="sbi-logs/fig_v3_vs_retrain.png")
    args = ap.parse_args()

    ref, ref_keys = load_run(args.ref)
    print(f"[ref] {args.ref}: keys={ref_keys}")

    # find the logM point-estimate and id/z columns robustly
    def get(d, candidates):
        for c in candidates:
            if c in d:
                return d[c]
        raise KeyError(f"none of {candidates} in {list(d.keys())}")

    m_ref = get(ref, ["logM_sbi", "logm_med", "logM_med", "mass_med"])
    ids_ref = get(ref, ["selected_indices", "ids", "object_id", "idx"])
    z_ref = get(ref, ["z", "zval", "z_spec", "redshift"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    zbins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    mbins = [7.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.5]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, run_dir in enumerate(args.run):
        run, run_keys = load_run(run_dir)
        m_run = get(run, ["logM_sbi", "logm_med", "logM_med", "mass_med"])
        ids_run = get(run, ["selected_indices", "ids", "object_id", "idx"])

        common, ia, ib = np.intersect1d(ids_ref, ids_run, return_indices=True)
        dm = m_run[ib] - m_ref[ia]
        zc = z_ref[ia]
        mc = m_ref[ia]
        label = os.path.basename(run_dir.rstrip("/"))
        print(f"\n[{label}] n_common={len(common)}")
        print(f"  delta(run - ref): median={np.median(dm):+.3f}  NMAD={nmad(dm):.3f}"
              f"  p16={np.percentile(dm,16):+.3f}  p84={np.percentile(dm,84):+.3f}")
        for lo, hi in zip(zbins[:-1], zbins[1:]):
            s = (zc >= lo) & (zc < hi)
            if s.sum() > 20:
                print(f"    z {lo:.1f}-{hi:.1f}: median={np.median(dm[s]):+.3f}  n={s.sum()}")
        for lo, hi in zip(mbins[:-1], mbins[1:]):
            s = (mc >= lo) & (mc < hi)
            if s.sum() > 20:
                print(f"    logM {lo:.1f}-{hi:.1f}: median={np.median(dm[s]):+.3f}  n={s.sum()}")

        c = colors[i]
        axes[0].hist(dm, bins=np.arange(-0.3, 0.61, 0.01), histtype="step",
                     color=c, label=f"{label}\nmed={np.median(dm):+.3f} nmad={nmad(dm):.3f}")
        # running median vs z
        zb = np.linspace(0, 5, 26)
        zmid, zmed = [], []
        for lo, hi in zip(zb[:-1], zb[1:]):
            s = (zc >= lo) & (zc < hi)
            if s.sum() > 30:
                zmid.append(0.5 * (lo + hi))
                zmed.append(np.median(dm[s]))
        axes[1].plot(zmid, zmed, "-o", ms=3, color=c, label=label)
        # running median vs mass
        mb = np.linspace(7.5, 11.5, 21)
        mmid, mmed = [], []
        for lo, hi in zip(mb[:-1], mb[1:]):
            s = (mc >= lo) & (mc < hi)
            if s.sum() > 30:
                mmid.append(0.5 * (lo + hi))
                mmed.append(np.median(dm[s]))
        axes[2].plot(mmid, mmed, "-o", ms=3, color=c, label=label)

    axes[0].set_xlabel(r"$\Delta \log M$ (retrain $-$ v3)")
    axes[0].set_ylabel("N")
    axes[0].axvline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("spec-z")
    axes[1].set_ylabel(r"median $\Delta \log M$")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylim(-0.1, 0.4)
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel(r"$\log M$ (v3)")
    axes[2].set_ylabel(r"median $\Delta \log M$")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylim(-0.1, 0.4)
    axes[2].legend(fontsize=7)
    fig.suptitle("Per-galaxy difference of retrained flows vs June-22 v3 (same input catalog)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
