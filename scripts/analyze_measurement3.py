#!/usr/bin/env python3
"""Generalized version of analyze_measurement2.py for measurement 3: any
number of named orders (sequential, in_degree, hop_distance,
traversal_frequency, ...) plus averaged random, H1 ratio computed against
the best-performing importance/tier-aware order among those evaluated, per
PREREGISTRATION.md's "may be chosen post hoc from among those" allowance.
Curves are built from the JSON records only.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_records(out_dir):
    return [json.loads(p.read_text()) for p in Path(out_dir).glob("*.json")]


def interpolate_crossing(bytes_arr, recall_arr, target):
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
    ap.add_argument("--baseline-orders", default="sequential", help="comma-separated, non-random baseline order names")
    ap.add_argument("--importance-orders", default="in_degree,hop_distance,traversal_frequency")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    seeds = [int(s) for s in args.random_seeds.split(",")]
    baseline_names = args.baseline_orders.split(",")
    importance_names = args.importance_orders.split(",")

    intact_candidates = [r["mean_recall"] for r in records if r["mean_absence_ceiling"] == 1.0]
    intact_recall = float(np.mean(intact_candidates))
    target_raw_recall = 0.9 * intact_recall
    print(f"intact recall (reference, n={len(intact_candidates)} runs): {intact_recall:.4f}")
    print(f"H1 target: 0.9 * intact = {target_raw_recall:.4f} raw recall@10\n")

    def curve_for(order_name):
        pts = [r for r in records if r["spec"]["order_description"] == order_name]
        pts.sort(key=lambda r: r["bytes"]["total"])
        return pts

    random_by_frac = {}
    for seed in seeds:
        for r in curve_for(f"random_seed{seed}"):
            key = len(r["spec"]["restored_groups"]["CODES"])
            random_by_frac.setdefault(key, []).append(r)
    random_curve = []
    for _, rs in sorted(random_by_frac.items()):
        random_curve.append({
            "bytes": {"total": float(np.mean([x["bytes"]["total"] for x in rs]))},
            "mean_recall": float(np.mean([x["mean_recall"] for x in rs])),
        })
    random_curve.sort(key=lambda r: r["bytes"]["total"])

    def bytes_recall(pts):
        return [r["bytes"]["total"] for r in pts], [r["mean_recall"] for r in pts]

    all_curves = {}
    for name in baseline_names:
        pts = curve_for(name)
        if pts:
            all_curves[name] = pts
    for name in importance_names:
        pts = curve_for(name)
        if pts:
            all_curves[name] = pts
    if random_curve:
        all_curves["random_avg"] = random_curve

    crossings = {}
    for name, pts in all_curves.items():
        b, r = bytes_recall(pts)
        crossings[name] = interpolate_crossing(b, r, target_raw_recall)
        n = len(pts)
        c = crossings[name]
        print(f"{name:22s} {n:3d} points, crossing bytes = {c if c is None else f'{c:.4e}'}")

    baseline_cross_candidates = {n: crossings[n] for n in baseline_names + ["random_avg"]
                                  if n in crossings and crossings[n] is not None}
    if not baseline_cross_candidates:
        print("\nno baseline (sequential/random) reached target in measured range -- cannot compute ratio")
        return
    baseline_name = min(baseline_cross_candidates, key=baseline_cross_candidates.get)
    baseline_cross = baseline_cross_candidates[baseline_name]
    print(f"\nbetter of {baseline_names + ['random']} needs {baseline_cross:.4e} bytes ({baseline_name})")

    importance_cross_candidates = {n: crossings[n] for n in importance_names
                                    if n in crossings and crossings[n] is not None}
    if not importance_cross_candidates:
        print("no importance/tier-aware order reached target in measured range")
        return
    best_name = min(importance_cross_candidates, key=importance_cross_candidates.get)
    best_cross = importance_cross_candidates[best_name]
    ratio = best_cross / baseline_cross
    print(f"best importance order: {best_name} needs {best_cross:.4e} bytes")
    print(f"H1 ratio (best-importance / better-of-baseline) = {ratio:.4f}  "
          f"({'PASSES' if ratio <= 1/3 else 'FAILS'} <= 1/3 threshold)")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "bytes_total", "mean_recall", "normalized_recall"])
        for name, pts in all_curves.items():
            if name == "random_avg":
                for r in pts:
                    w.writerow(["random_avg", r["bytes"]["total"], r["mean_recall"],
                                r["mean_recall"] / intact_recall])
            else:
                for r in pts:
                    w.writerow([name, r["bytes"]["total"], r["mean_recall"],
                                r["mean_recall"] / intact_recall])
    print(f"\nwrote curve data to {args.out_csv}")


if __name__ == "__main__":
    main()
