#!/usr/bin/env python3
"""HARNESS_JOBS.md measurement 2 (the H1 read): SIFT10M, random vs.
in-degree order (plus sequential, needed as the other half of
PREREGISTRATION.md's "better of sequential and random" baseline), all three
tiers restored together, evaluated on the held-out eval query subset.

Query set: uses the same 5K/5K workload/eval split reserved for the
traversal-frequency proxy (harness/importance.py: split_workload_eval_queries,
seed=1234), evaluating only on the 5K eval half, even though this
measurement doesn't use the workload half itself -- so every later
measurement's curve is comparable against the same eval set.

Coarse-then-refine: pass a --fractions list; already-computed (mask, order,
seed) points are skipped via runner.py's resumability, so refining is just
calling this again with additional fractions.

No hard-coded paths: every input is a CLI argument.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points, read_bin_header
from harness.mask import make_rank_grouping, compute_group_byte_stats, mask_from_fraction, TIERS
from harness.importance import (
    compute_in_degree, compute_hop_distance, order_by_score, split_workload_eval_queries,
)
from harness.absence import read_ground_truth
from harness.runner import RunSpec, run_masked_sweep_point


def read_query_file(path, dtype):
    n, dim = read_bin_header(path)
    data = np.fromfile(path, dtype=dtype, count=n * dim, offset=8)
    return data.reshape(n, dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-name", default="sift10m")
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
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query-split-seed", type=int, default=1234)
    args = ap.parse_args()

    dtype = np.uint8 if args.vector_dtype == "uint8" else np.float32
    fractions = [float(x) for x in args.fractions.split(",")]
    random_seeds = [int(x) for x in args.random_seeds.split(",")]
    if len(random_seeds) < 5:
        raise ValueError("PREREGISTRATION.md requires >=5 random seeds")

    print(f"[{args.corpus_name}] loading index...")
    idx = DiskIndex(args.disk_index, dtype)
    pqt = PQTable(args.pq_pivots)
    pqc = PQCodes(args.pq_compressed)
    ep = read_entry_points(args.disk_index, idx)
    npts = idx.meta.npts

    all_queries = read_query_file(args.query_file, dtype)
    gt_ids, _ = read_ground_truth(args.ground_truth)
    workload_ids, eval_ids = split_workload_eval_queries(
        all_queries.shape[0], seed=args.query_split_seed
    )
    eval_queries = all_queries[eval_ids]
    eval_gt = gt_ids[eval_ids]
    print(f"[{args.corpus_name}] eval query subset: {eval_queries.shape[0]} "
          f"(workload-reserved: {workload_ids.shape[0]})")

    search_params = dict(L=args.L, W=args.W, k=args.k, io_limit=None)

    print(f"[{args.corpus_name}] computing in-degree proxy...")
    in_degree = compute_in_degree(idx)

    if ep.node_ids.shape[0] != 1:
        raise NotImplementedError(
            f"{args.corpus_name} has {ep.node_ids.shape[0]} candidate entry points; "
            "this script's single hop_distance_from_entry fallback array only covers "
            "the single-medoid case (SIFT10M, Deep10M). MS MARCO needs a per-entry-point "
            "hop-distance table before this script can be reused for it."
        )
    print(f"[{args.corpus_name}] computing hop-distance-from-entry proxy "
          f"(also used as the CODES-fallback-seed distance under masking)...")
    hop_distance_from_entry = compute_hop_distance(idx, int(ep.node_ids[0]))

    orders = {}
    orders["sequential"] = (np.arange(npts, dtype=np.int64), None)
    orders["in_degree"] = (order_by_score(in_degree, descending=True), None)
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
                label=f"measurement2 order={order_name} frac={frac}",
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

    print(f"[{args.corpus_name}] measurement 2 sweep complete: {total_runs} points in "
          f"{(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
