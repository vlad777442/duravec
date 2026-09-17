#!/usr/bin/env python3
"""Plot the measurement-2 restoration curve from the CSV produced by
analyze_measurement2.py. Regenerated from the CSV only (HARNESS_JOBS.md 8:
plots are regenerated from the JSON records by a script -- the CSV is that
script's intermediate, itself regenerable from the JSON records)."""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--title", default="SIFT10M restoration curve (measurement 2)")
    args = ap.parse_args()

    series = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            series[row["order"]].append(
                (float(row["bytes_total"]), float(row["normalized_recall"]))
            )
    for k in series:
        series[k].sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "sequential": dict(color="#888888", marker="o", label="sequential"),
        "random_avg5seeds": dict(color="#d95f02", marker="s", label="random (avg of 5 seeds)"),
        "random_avg": dict(color="#d95f02", marker="s", label="random (avg of 5 seeds)"),
        "in_degree": dict(color="#1b9e77", marker="^", label="in-degree"),
        "hop_distance": dict(color="#7570b3", marker="v", label="hop-distance"),
        "traversal_frequency": dict(color="#e7298a", marker="d", label="traversal-frequency"),
    }
    for name, pts in series.items():
        xs = [p[0] / 1e9 for p in pts]
        ys = [p[1] for p in pts]
        style = styles.get(name, dict(label=name))
        ax.plot(xs, ys, **style)

    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="H1 target (0.9 normalized recall)")
    ax.set_xlabel("Bytes restored (GB, actual content bytes)")
    ax.set_ylabel("Normalized recall@10")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
