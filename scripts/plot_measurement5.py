#!/usr/bin/env python3
"""Plot measurement 5's B5 sweep: broken-navigation (absence_ceiling - recall)
vs L and vs W, faceted by restored fraction, one line per corpus. Reads the
JSON records directly."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results_root, corpus):
    recs = [json.loads(p.read_text()) for p in (Path(results_root) / corpus).glob("*.json")]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    corpora = [("sift10m", "SIFT10M", "#1b9e77"), ("deep10m", "Deep10M", "#d95f02"),
               ("msmarco", "MS MARCO", "#7570b3")]
    fractions = [0.02, 0.05, 0.1]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")

    for corpus_key, corpus_label, color in corpora:
        recs = load(args.results_root, corpus_key)
        # bucket by fraction (number of restored CODES groups is a stable proxy)
        for r in recs:
            n_groups_restored = len(r["spec"]["restored_groups"]["CODES"])
            r["_frac_key"] = n_groups_restored
        frac_keys = sorted(set(r["_frac_key"] for r in recs))
        frac_label_map = dict(zip(sorted(frac_keys), fractions)) if len(frac_keys) == 3 else None

        for col, frac in enumerate(fractions):
            target_key = frac_keys[fractions.index(frac)] if frac_label_map else None
            subset = [r for r in recs if r["_frac_key"] == target_key]

            l_series = sorted(
                (r["spec"]["search_params"]["L"], r["mean_absence_ceiling"] - r["mean_recall"])
                for r in subset if r["spec"]["search_params"]["W"] == 4
            )
            w_series = sorted(
                (r["spec"]["search_params"]["W"], r["mean_absence_ceiling"] - r["mean_recall"])
                for r in subset if r["spec"]["search_params"]["L"] == 50
            )

            ax_l = axes[0, col]
            ax_w = axes[1, col]
            if l_series:
                xs, ys = zip(*l_series)
                ax_l.plot(xs, ys, marker="o", color=color, label=corpus_label)
            if w_series:
                xs, ys = zip(*w_series)
                ax_w.plot(xs, ys, marker="s", color=color, label=corpus_label)

            ax_l.set_title(f"frac={frac}")
            ax_l.set_xlabel("L")
            ax_w.set_xlabel("W")

    axes[0, 0].set_ylabel("Broken navigation\n(ceiling - recall)")
    axes[1, 0].set_ylabel("Broken navigation\n(ceiling - recall)")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("B5: search effort vs. broken navigation (measurement 5)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
