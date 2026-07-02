"""
Merge parallel atlas shards (.dbatlas) into one atlas for training.

Atlas generation (validate_noise_model.py --simulate-only) is single-threaded FSPS,
so to use many CPU cores you launch K shards in parallel, each with a DISTINCT --seed
and --atlas-name, then merge them here into a single atlas whose N = sum of shard N.

Each shard file is a hickle dict with row-aligned arrays:
  zval, sfh_tuple, mstar, sfr, dust, met, sed   (axis 0 = galaxy)

Example (16 shards of 3125 -> 50k, in library/):
  merged = atlas_euclid_total_v1_50000_Nparam_2.dbatlas
  python examples/merge_atlas.py \
      --shards library/atlas_v1_shard*_3125_Nparam_2.dbatlas \
      --out-name atlas_euclid_total_v1 --path library --nparam 2
Then train with:  --atlas-name atlas_euclid_total_v1 --n-sim <total N printed>
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import hickle

KEYS = ["zval", "sfh_tuple", "mstar", "sfr", "dust", "met", "sed"]


def main():
    ap = argparse.ArgumentParser(description="Merge dense_basis atlas shards into one")
    ap.add_argument("--shards", nargs="+", required=True,
                    help="Shard .dbatlas files (globs allowed).")
    ap.add_argument("--out-name", required=True,
                    help="Output atlas stem (train --atlas-name), N is appended automatically.")
    ap.add_argument("--path", default="library", help="Output directory (default: library)")
    ap.add_argument("--nparam", type=int, default=2, help="Nparam in filename (parametric=2)")
    args = ap.parse_args()

    # expand globs
    files = []
    for s in args.shards:
        hits = sorted(glob.glob(s))
        files.extend(hits if hits else [s])
    files = [f for f in files if Path(f).exists()]
    if not files:
        raise SystemExit("No shard files found.")
    print(f"Merging {len(files)} shards:")

    parts = {k: [] for k in KEYS}
    total = 0
    for f in files:
        d = hickle.load(f)
        n = len(np.asarray(d["mstar"]))
        total += n
        print(f"  {f}  (N={n})")
        for k in KEYS:
            parts[k].append(np.asarray(d[k]))

    merged = {k: np.concatenate(parts[k], axis=0) for k in KEYS}
    # sanity: all keys same length
    lengths = {k: len(merged[k]) for k in KEYS}
    assert len(set(lengths.values())) == 1, f"row mismatch after merge: {lengths}"

    outdir = Path(args.path)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.out_name}_{total}_Nparam_{args.nparam}.dbatlas"
    try:
        hickle.dump(merged, str(out), compression="gzip", compression_opts=9)
    except Exception:
        hickle.dump(merged, str(out))
    print(f"\nwrote {out}  (total N={total})")
    print(f"logM range [{np.nanmin(merged['mstar']):.2f}, {np.nanmax(merged['mstar']):.2f}], "
          f"z range [{np.nanmin(merged['zval']):.2f}, {np.nanmax(merged['zval']):.2f}]")
    print(f"\nTrain with:  --atlas-name {args.out_name} --n-sim {total}")


if __name__ == "__main__":
    main()
