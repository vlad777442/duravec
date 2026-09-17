#!/usr/bin/env python3
"""HARNESS_JOBS.md measurement 4: tier curves. Codes-first, then
codes+adjacency, then vectors -- three cumulative stages restoring each
tier at 100% across the whole graph, isolating each tier's marginal
contribution to recall (as opposed to measurements 2/3, which restore all
three tiers together for a growing subset of nodes chosen by importance
order). Expectation from HARNESS_JOBS.md 7.4: a large recall jump from
unlocking VEC on MS MARCO (88% of bytes are vectors), and very little from
unlocking VEC on SIFT10M (23% of bytes are vectors, adjacency dominates at
72%) -- and if SIFT shows a large VEC-tier gain instead, that means the
harness is wrong, not that the data is surprising.

Run after H1 already failed on all three corpora (measurement 3): this
measurement is being run for the write-up's diagnostic value, not as a
go/no-go gate (PREREGISTRATION.md 2: once H1 fails, tier curves don't
change the outcome, but HARNESS_JOBS.md 4 keeps them as characterization).

Uses the same 5,000-query eval subset as measurements 2/3 (seed 1234) for
comparability. A single-group grouping is enough here since every mask is
either "restore this tier everywhere" or "restore it nowhere" -- there is
no per-node fraction to sweep.

No hard-coded paths: every input is a CLI argument.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points, read_bin_header
from harness.mask import make_rank_grouping, compute_group_byte_stats, mask_from_groups
from harness.importance import split_workload_eval_queries
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
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query-split-seed", type=int, default=1234)
    args = ap.parse_args()

    dtype = np.uint8 if args.vector_dtype == "uint8" else np.float32

    print(f"[{args.corpus_name}] loading index...")
    idx = DiskIndex(args.disk_index, dtype)
    pqt = PQTable(args.pq_pivots)
    pqc = PQCodes(args.pq_compressed)
    ep = read_entry_points(args.disk_index, idx)
    npts = idx.meta.npts

    all_queries = read_query_file(args.query_file, dtype)
    gt_ids, _ = read_ground_truth(args.ground_truth)
    _, eval_ids = split_workload_eval_queries(all_queries.shape[0], seed=args.query_split_seed)
    eval_queries = all_queries[eval_ids]
    eval_gt = gt_ids[eval_ids]
    print(f"[{args.corpus_name}] eval query subset: {eval_queries.shape[0]}")

    search_params = dict(L=args.L, W=args.W, k=args.k, io_limit=None)

    grouping = make_rank_grouping("all_nodes", np.arange(npts, dtype=np.int64), npts, "all_nodes_single_group")
    byte_stats = compute_group_byte_stats(grouping, idx)
    print(f"[{args.corpus_name}] byte totals: CODES={byte_stats.codes_bytes[0]:.3e} "
          f"ADJ={byte_stats.adj_bytes_actual[0]:.3e} VEC={byte_stats.vec_bytes[0]:.3e}")

    stages = [
        ("codes_only", {"CODES": [0], "ADJ": [], "VEC": []}),
        ("codes_adjacency", {"CODES": [0], "ADJ": [0], "VEC": []}),
        ("intact", {"CODES": [0], "ADJ": [0], "VEC": [0]}),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for stage_name, groups_by_tier in stages:
        mask = mask_from_groups(grouping, groups_by_tier)
        restored_groups = {t: sorted(mask.restored_groups[t]) for t in mask.restored_groups}
        spec = RunSpec(
            corpus=args.corpus_name,
            grouping_name=grouping.name,
            order_description=f"tier_stage_{stage_name}",
            restored_groups=restored_groups,
            search_params=search_params,
            adj_byte_convention="actual",
            label=f"measurement4 stage={stage_name}",
        )
        record = run_masked_sweep_point(
            spec, str(out_dir), mask, grouping, byte_stats, idx, pqt, pqc, ep,
            eval_queries, eval_gt,
        )
        results[stage_name] = record
        print(f"[{args.corpus_name}] stage={stage_name:16s} bytes={record['bytes']['total']:.3e} "
              f"mean_recall={record['mean_recall']:.4f} "
              f"mean_absence_ceiling={record['mean_absence_ceiling']:.4f} "
              f"miss_rate={np.mean(record['per_query']['miss_count']):.1f} "
              f"code_only_rate={np.mean(record['per_query']['code_only_count']):.1f}")

    intact = results["intact"]["mean_recall"]
    codes_only = results["codes_only"]["mean_recall"]
    codes_adj = results["codes_adjacency"]["mean_recall"]
    print(f"\n[{args.corpus_name}] TIER CONTRIBUTIONS")
    print(f"  codes-only recall:        {codes_only:.4f}")
    print(f"  + adjacency (Delta_ADJ):  {codes_adj - codes_only:+.4f}  -> {codes_adj:.4f}")
    print(f"  + vectors   (Delta_VEC):  {intact - codes_adj:+.4f}  -> {intact:.4f}")


if __name__ == "__main__":
    main()
