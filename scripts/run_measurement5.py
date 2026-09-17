#!/usr/bin/env python3
"""HARNESS_JOBS.md measurement 5: the B5 beam-width sweep, at several
restored fractions, with the absence/navigation decomposition from 5.3.

Both L (search-list/candidate-list capacity) and W (beam width: candidates
expanded per hop) are swept, at the same restored fractions, because both
turn out to matter -- an earlier assumption while building harness/search.py
was that W only batches I/O and has no effect on which nodes end up
expanded, since the algorithm runs to exhaustion of the L-bounded candidate
list regardless of batch size. That assumption was wrong, and this
script's own first run caught it: recall was not flat across W. The
mechanism (worth stating since it wasn't obvious going in): the candidate
list is a capacity-L bounded structure where already-expanded entries still
occupy a slot until evicted by a strictly better later insertion. A larger
W expands several close-but-not-yet-reconsidered candidates together off a
single shared snapshot of the list before any of their discoveries get
folded back in, which is exactly the classical beam-search-quality
mechanism (wider beams explore more broadly before committing) -- so W is
a genuine, if secondary, search-effort knob after all, not just an I/O
parallelism parameter. L's effect is larger in absolute terms at every
fraction tested, but W's effect is real, not noise.

Uses a documented, deliberate subsample of the eval query set (default
1000 of the 5000) -- this is a characterization sweep, not a go/no-go gate
(H1 already failed on all three corpora; B5 doesn't change that outcome
either way per PREREGISTRATION.md 2/4), so the full 5000-query cost isn't
warranted. Flagged here rather than silently done.

No hard-coded paths: every input is a CLI argument.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points, read_bin_header
from harness.mask import make_rank_grouping, compute_group_byte_stats, mask_from_fraction, TIERS
from harness.importance import compute_hop_distance, split_workload_eval_queries
from harness.absence import read_ground_truth
from harness.runner import RunSpec, run_masked_sweep_point


def read_query_file(path, dtype):
    n, dim = read_bin_header(path)
    data = np.fromfile(path, dtype=dtype, count=n * dim, offset=8)
    return data.reshape(n, dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--disk-index", required=True)
    ap.add_argument("--pq-pivots", required=True)
    ap.add_argument("--pq-compressed", required=True)
    ap.add_argument("--query-file", required=True)
    ap.add_argument("--vector-dtype", required=True, choices=["uint8", "float32"])
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-size", type=int, default=1000)
    ap.add_argument("--fractions", default="0.02,0.05,0.1")
    ap.add_argument("--L-values", default="50,100,200,400")
    ap.add_argument("--W-values", default="2,4,8,16")
    ap.add_argument("--random-seed", type=int, default=101)
    ap.add_argument("--num-queries", type=int, default=1000,
                     help="deliberate subsample of the 5000-query eval set (see module docstring)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query-split-seed", type=int, default=1234)
    args = ap.parse_args()

    dtype = np.uint8 if args.vector_dtype == "uint8" else np.float32
    fractions = [float(x) for x in args.fractions.split(",")]
    L_values = [int(x) for x in args.L_values.split(",")]
    W_values = [int(x) for x in args.W_values.split(",")]

    print(f"[{args.corpus_name}] loading index...")
    idx = DiskIndex(args.disk_index, dtype)
    pqt = PQTable(args.pq_pivots)
    pqc = PQCodes(args.pq_compressed)
    ep = read_entry_points(args.disk_index, idx)
    npts = idx.meta.npts

    all_queries = read_query_file(args.query_file, dtype)
    gt_ids, _ = read_ground_truth(args.ground_truth)
    _, eval_ids = split_workload_eval_queries(all_queries.shape[0], seed=args.query_split_seed)
    eval_queries = all_queries[eval_ids][: args.num_queries]
    eval_gt = gt_ids[eval_ids][: args.num_queries]
    print(f"[{args.corpus_name}] B5 query subsample: {eval_queries.shape[0]} "
          f"(of {eval_ids.shape[0]}-query eval set)")

    hop_distance_from_entry = compute_hop_distance(idx, ep.node_ids)

    rng = np.random.default_rng(args.random_seed)
    order = rng.permutation(npts)
    grouping = make_rank_grouping(f"random_seed{args.random_seed}", order, args.group_size, "random")
    byte_stats = compute_group_byte_stats(grouping, idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.corpus_name}] === B5 sweep: L vs restored fraction (W fixed at stock 4) ===")
    for frac in fractions:
        mask = mask_from_fraction(grouping, frac, tiers=TIERS)
        restored_groups = {t: sorted(mask.restored_groups[t]) for t in mask.restored_groups}
        for L in L_values:
            spec = RunSpec(
                corpus=args.corpus_name, grouping_name=grouping.name,
                order_description=grouping.order_description, restored_groups=restored_groups,
                search_params=dict(L=L, W=4, k=args.k, io_limit=None), adj_byte_convention="actual",
                seed=args.random_seed, label=f"measurement5 frac={frac} L={L}",
            )
            record = run_masked_sweep_point(
                spec, str(out_dir), mask, grouping, byte_stats, idx, pqt, pqc, ep,
                eval_queries, eval_gt, hop_distance_from_entry=hop_distance_from_entry,
            )
            ceiling = record["mean_absence_ceiling"]
            recall = record["mean_recall"]
            broken_nav = ceiling - recall
            print(f"  frac={frac:5.2f} L={L:4d}  recall={recall:.4f}  ceiling={ceiling:.4f}  "
                  f"broken_navigation={broken_nav:+.4f}  mean_hops={np.mean(record['per_query']['hops']):.1f} "
                  f"wall={record['wall_clock_sec']:.1f}s")

    print(f"\n[{args.corpus_name}] === B5 sweep: W vs restored fraction (L fixed at stock 50) ===")
    for frac in fractions:
        mask = mask_from_fraction(grouping, frac, tiers=TIERS)
        restored_groups = {t: sorted(mask.restored_groups[t]) for t in mask.restored_groups}
        for W in W_values:
            spec = RunSpec(
                corpus=args.corpus_name, grouping_name=grouping.name,
                order_description=grouping.order_description, restored_groups=restored_groups,
                search_params=dict(L=50, W=W, k=args.k, io_limit=None), adj_byte_convention="actual",
                seed=args.random_seed, label=f"measurement5 frac={frac} W={W}",
            )
            record = run_masked_sweep_point(
                spec, str(out_dir), mask, grouping, byte_stats, idx, pqt, pqc, ep,
                eval_queries, eval_gt, hop_distance_from_entry=hop_distance_from_entry,
            )
            ceiling = record["mean_absence_ceiling"]
            recall = record["mean_recall"]
            broken_nav = ceiling - recall
            print(f"  frac={frac:5.2f} W={W:4d}  recall={recall:.4f}  ceiling={ceiling:.4f}  "
                  f"broken_navigation={broken_nav:+.4f}  mean_hops={np.mean(record['per_query']['hops']):.1f} "
                  f"wall={record['wall_clock_sec']:.1f}s")


if __name__ == "__main__":
    main()
