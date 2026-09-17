#!/usr/bin/env python3
"""Plot measurement 4's tier contributions (codes-only / +adjacency /
+vectors) as a grouped bar chart across corpora, reading directly from the
JSON records written by run_measurement4.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_stage_recalls(results_dir):
    recs = [json.loads(p.read_text()) for p in Path(results_dir).glob("*.json")]
    by_stage = {r["spec"]["order_description"].replace("tier_stage_", ""): r["mean_recall"] for r in recs}
    return by_stage["codes_only"], by_stage["codes_adjacency"], by_stage["intact"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="dir containing sift10m/, deep10m/, msmarco/ subdirs")
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    corpora = ["sift10m", "deep10m", "msmarco"]
    labels = ["SIFT10M", "Deep10M", "MS MARCO"]
    codes_only, codes_adj, intact = [], [], []
    for c in corpora:
        co, ca, it = load_stage_recalls(Path(args.results_root) / c)
        codes_only.append(co)
        codes_adj.append(ca)
        intact.append(it)

    x = np.arange(len(corpora))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 5))
    codes_only = np.array(codes_only)
    codes_adj = np.array(codes_adj)
    intact = np.array(intact)

    ax.bar(x, codes_only, width, label="codes-only", color="#cccccc")
    ax.bar(x, codes_adj - codes_only, width, bottom=codes_only, label="+ adjacency", color="#1b9e77")
    ax.bar(x, intact - codes_adj, width, bottom=codes_adj, label="+ vectors", color="#d95f02")

    for i, (ca, it) in enumerate(zip(codes_adj, intact)):
        ax.text(i, ca / 2, f"{ca:.3f}", ha="center", va="center", fontsize=9)
        ax.text(i, ca + (it - ca) / 2, f"+{it - ca:.3f}", ha="center", va="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("Tier contributions to recall (measurement 4)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
