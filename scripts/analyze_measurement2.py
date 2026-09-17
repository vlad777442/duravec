#!/usr/bin/env python3
"""Build the measurement-2 restoration curve from JSON provenance records
and locate where each order crosses 0.9 normalized recall@10, per
PREREGISTRATION.md H1. Curves are built from the records only, never from
in-memory sweep state (HARNESS_JOBS.md 5.5/8).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_records(out_dir):
    records = []
    for p in Path(out_dir).glob("*.json"):
        records.append(json.loads(p.read_text()))
    return records


def curve_for_order(records, order_prefix, exact=False):
    pts = [
        r for r in records
        if (r["spec"]["order_description"] == order_prefix if exact
            else r["spec"]["grouping_name"].startswith(order_prefix))
    ]
    pts.sort(key=lambda r: r["bytes"]["total"])
    return pts


def interpolate_crossing(bytes_arr, recall_arr, target):
    """First-crossing byte value where recall_arr reaches target, linear
    interpolation between the bracketing measured points. Returns None if
    target is never reached in the measured range."""
    for i in range(1, len(bytes_arr)):
        if recall_arr[i - 1] < target <= recall_arr[i]:
            b0, b1 = bytes_arr[i - 1], bytes_arr[i]
            r0, r1 = recall_arr[i - 1], recall_arr[i]
            frac = (target - r0) / (r1 - r0)
            return b0 + frac * (b1 - b0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--random-seeds", default="101,102,103,104,105")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    seeds = [int(s) for s in args.random_seeds.split(",")]

    # intact recall: any order at frac giving mean_absence_ceiling == 1.0 (fully restored)
    intact_candidates = [r["mean_recall"] for r in records if r["mean_absence_ceiling"] == 1.0]
    intact_recall = float(np.mean(intact_candidates))
    target_raw_recall = 0.9 * intact_recall
    print(f"intact recall (reference, n={len(intact_candidates)} runs): {intact_recall:.4f}")
    print(f"H1 target: 0.9 * intact = {target_raw_recall:.4f} raw recall@10\n")

    seq_pts = curve_for_order(records, "sequential", exact=True)
    ind_pts = curve_for_order(records, "in_degree", exact=True)

    # random: average recall across seeds at each matching fraction
    random_by_frac = {}
    for seed in seeds:
        pts = curve_for_order(records, f"random_seed{seed}", exact=True)
        for r in pts:
            frac = tuple(sorted(r["spec"]["restored_groups"]["CODES"]))
            frac_len = len(frac)
            random_by_frac.setdefault(frac_len, []).append(r)

    random_curve = []
    for frac_len, rs in sorted(random_by_frac.items()):
        mean_bytes = float(np.mean([r["bytes"]["total"] for r in rs]))
        mean_recall = float(np.mean([r["mean_recall"] for r in rs]))
        random_curve.append({"bytes": {"total": mean_bytes}, "mean_recall": mean_recall, "n_seeds": len(rs)})
    random_curve.sort(key=lambda r: r["bytes"]["total"])

    def bytes_recall(pts):
        return [r["bytes"]["total"] for r in pts], [r["mean_recall"] for r in pts]

    seq_b, seq_r = bytes_recall(seq_pts)
    ind_b, ind_r = bytes_recall(ind_pts)
    rand_b, rand_r = bytes_recall(random_curve)

    seq_cross = interpolate_crossing(seq_b, seq_r, target_raw_recall)
    ind_cross = interpolate_crossing(ind_b, ind_r, target_raw_recall)
    rand_cross = interpolate_crossing(rand_b, rand_r, target_raw_recall)

    print(f"sequential: {len(seq_pts)} points, crossing bytes = {seq_cross}")
    print(f"in_degree:  {len(ind_pts)} points, crossing bytes = {ind_cross}")
    print(f"random (avg of {len(seeds)} seeds): {len(random_curve)} points, crossing bytes = {rand_cross}")

    baseline_cross = min(x for x in [seq_cross, rand_cross] if x is not None)
    baseline_name = "sequential" if baseline_cross == seq_cross else "random"
    print(f"\nbetter of sequential/random needs {baseline_cross:.4e} bytes "
          f"({baseline_name})")
    if ind_cross is not None:
        ratio = ind_cross / baseline_cross
        print(f"in_degree needs {ind_cross:.4e} bytes")
        print(f"H1 ratio (in_degree / better-of-seq-random) = {ratio:.4f}  "
              f"({'PASSES' if ratio <= 1/3 else 'FAILS'} <= 1/3 threshold)")
    else:
        print("in_degree never reached target in measured range")

    # write full curve data as CSV, all points, for plotting/re-derivation
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "bytes_total", "mean_recall", "normalized_recall", "mean_absence_ceiling"])
        for name, pts in [("sequential", seq_pts), ("in_degree", ind_pts)]:
            for r in pts:
                w.writerow([name, r["bytes"]["total"], r["mean_recall"],
                            r["mean_recall"] / intact_recall, r["mean_absence_ceiling"]])
        for r in random_curve:
            w.writerow(["random_avg5seeds", r["bytes"]["total"], r["mean_recall"],
                        r["mean_recall"] / intact_recall, ""])

    print(f"\nwrote curve data to {args.out_csv}")


if __name__ == "__main__":
    main()
