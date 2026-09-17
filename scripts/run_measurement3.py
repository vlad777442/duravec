#!/usr/bin/env python3
"""HARNESS_JOBS.md measurement 3: extend measurement 2 to Deep10M and MS
MARCO, same orders (sequential, random x>=5 seeds, in-degree), plus add
hop-distance and traversal-frequency orders. Generalizes run_measurement2.py
to handle multiple candidate entry points (MS MARCO's sharded build has 3;
SIFT10M/Deep10M have 1) via multi-source hop distance, and to include this
run's own traversal-frequency workload replay.

Re-running this for a corpus that measurement 2 already swept (SIFT10M) is
intentional and cheap: RunSpec content-hashes are identical for the three
orders already computed there, so run_masked_sweep_point's resumability
skips them and only computes the two new orders.

Query set: same 5K/5K workload/eval split as measurement 2 (seed=1234).
Traversal-frequency is computed by replaying the workload half at this
run's own search params (HARNESS_JOBS.md 6: "This is what a warm-up policy
would use" -- using the same stock params as the recall measurement itself
keeps it representative of the actual serving configuration).

No hard-coded paths: every input is a CLI argument.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points, read_bin_header
from harness.mask import make_rank_grouping, compute_group_byte_stats, mask_from_fraction, TIERS
from harness.importance import (
    compute_in_degree, compute_hop_distance, order_by_score,
    split_workload_eval_queries, compute_traversal_frequency,
)
from harness.absence import read_ground_truth
from harness.runner import RunSpec, run_masked_sweep_point
from harness.search import SearchParams


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
    ap.add_argument("--fractions", required=True, help="comma-separated, e.g. 0,0.05,0.1,1.0")
    ap.add_argument("--random-seeds", default="101,102,103,104,105")
    ap.add_argument("--orders", default="sequential,in_degree,hop_distance,traversal_frequency,random",
                     help="comma-separated subset to run this invocation")
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query-split-seed", type=int, default=1234)
    ap.add_argument("--traversal-tie-break-seed", type=int, default=777)
    args = ap.parse_args()

    dtype = np.uint8 if args.vector_dtype == "uint8" else np.float32
    fractions = [float(x) for x in args.fractions.split(",")]
    random_seeds = [int(x) for x in args.random_seeds.split(",")]
    if len(random_seeds) < 5:
        raise ValueError("PREREGISTRATION.md requires >=5 random seeds")
    wanted_orders = set(args.orders.split(","))

    print(f"[{args.corpus_name}] loading index...")
    idx = DiskIndex(args.disk_index, dtype)
    pqt = PQTable(args.pq_pivots)
    pqc = PQCodes(args.pq_compressed)
    ep = read_entry_points(args.disk_index, idx)
    npts = idx.meta.npts
    print(f"[{args.corpus_name}] {ep.node_ids.shape[0]} candidate entry point(s): "
          f"{ep.node_ids.tolist()}")

    all_queries = read_query_file(args.query_file, dtype)
    gt_ids, _ = read_ground_truth(args.ground_truth)
    workload_ids, eval_ids = split_workload_eval_queries(
        all_queries.shape[0], seed=args.query_split_seed
    )
    eval_queries = all_queries[eval_ids]
    eval_gt = gt_ids[eval_ids]
    workload_queries = all_queries[workload_ids]
    print(f"[{args.corpus_name}] eval query subset: {eval_queries.shape[0]} "
          f"(workload: {workload_queries.shape[0]})")

    search_params = dict(L=args.L, W=args.W, k=args.k, io_limit=None)

    print(f"[{args.corpus_name}] computing hop-distance-from-nearest-entry-point "
          f"proxy (also the CODES-fallback-seed distance under masking)...")
    hop_distance_from_entry = compute_hop_distance(idx, ep.node_ids)

    orders = {}
    if "sequential" in wanted_orders:
        orders["sequential"] = (np.arange(npts, dtype=np.int64), None)
    if "in_degree" in wanted_orders:
        print(f"[{args.corpus_name}] computing in-degree proxy...")
        in_degree = compute_in_degree(idx)
        orders["in_degree"] = (order_by_score(in_degree, descending=True), None)
    if "hop_distance" in wanted_orders:
        orders["hop_distance"] = (
            order_by_score(hop_distance_from_entry, descending=False), None
        )
    if "traversal_frequency" in wanted_orders:
        print(f"[{args.corpus_name}] computing traversal-frequency proxy "
              f"({workload_queries.shape[0]}-query workload replay)...")
        visit_count, never_visited_frac, tie_key = compute_traversal_frequency(
            idx, pqt, pqc, ep, workload_queries, SearchParams(**search_params),
            tie_break_seed=args.traversal_tie_break_seed,
        )
        print(f"[{args.corpus_name}] traversal-frequency: {never_visited_frac:.4f} "
              f"of nodes never visited by the workload")
        orders["traversal_frequency"] = (
            order_by_score(visit_count, descending=True, tie_break_key=tie_key), None
        )
    if "random" in wanted_orders:
        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            orders[f"random_seed{seed}"] = (rng.permutation(npts), seed)

    groupings = {}
    byte_stats = {}
    for name, (order, _seed) in orders.items():
        g = make_rank_grouping(name, order, args.group_size, name)
        groupings[name] = g
        byte_stats[name] = compute_group_byte_stats(g, idx)
        print(f"[{args.corpus_name}] grouping {name}: {g.n_groups} groups")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(orders) * len(fractions)
    done = 0
    t_start = time.time()
    for order_name, (order, seed) in orders.items():
        grouping = groupings[order_name]
        bstats = byte_stats[order_name]
        for frac in fractions:
            mask = mask_from_fraction(grouping, frac, tiers=TIERS)
            restored_groups = {t: sorted(mask.restored_groups[t]) for t in mask.restored_groups}
            spec = RunSpec(
                corpus=args.corpus_name,
                grouping_name=grouping.name,
                order_description=grouping.order_description,
                restored_groups=restored_groups,
                search_params=search_params,
                adj_byte_convention="actual",
                seed=seed,
                label=f"measurement3 order={order_name} frac={frac}",
            )
            t0 = time.time()
            record = run_masked_sweep_point(
                spec, str(out_dir), mask, grouping, bstats, idx, pqt, pqc, ep,
                eval_queries, eval_gt, hop_distance_from_entry=hop_distance_from_entry,
            )
            done += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            print(f"[{args.corpus_name}] ({done}/{total_runs}) order={order_name} frac={frac} "
                  f"bytes={record['bytes']['total']:.3e} mean_recall={record['mean_recall']:.4f} "
                  f"mean_absence_ceiling={record['mean_absence_ceiling']:.4f} "
                  f"run_time={elapsed:.1f}s total_elapsed={total_elapsed / 60:.1f}min "
                  f"{'(cached)' if elapsed < 0.1 else ''}")

    print(f"[{args.corpus_name}] measurement 3 sweep complete: {total_runs} points in "
          f"{(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
