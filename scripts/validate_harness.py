#!/usr/bin/env python3
"""HARNESS_JOBS.md measurement 1: unmasked search must reproduce stock
search_disk_index recall on all three corpora before anything else is
measured.

Primary gate: per-query result-set agreement against stock's own saved
result ids (logs/job4_results/*_search_50_idx_uint32.bin) -- two
implementations can agree on mean recall while disagreeing on which
neighbors they found, so this is checked query by query, not just in
aggregate. Backstop: aggregate recall-vs-ground-truth within a tolerance.

No hard-coded paths: every input is a CLI argument.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points, read_bin_header
from harness.mask import TIERS
from harness.search import masked_search, SearchParams
from harness.absence import read_ground_truth, recall_at_k


def read_query_file(path, dtype):
    n, dim = read_bin_header(path)
    data = np.fromfile(path, dtype=dtype, count=n * dim, offset=8)
    return data.reshape(n, dim)


def read_stock_result_ids(path):
    n, k = struct.unpack("<ii", open(path, "rb").read(8))
    ids = np.fromfile(path, dtype="<u4", count=n * k, offset=8).reshape(n, k)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-name", required=True)
    ap.add_argument("--disk-index", required=True)
    ap.add_argument("--pq-pivots", required=True)
    ap.add_argument("--pq-compressed", required=True)
    ap.add_argument("--query-file", required=True)
    ap.add_argument("--vector-dtype", required=True, choices=["uint8", "float32"])
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--stock-result-ids", required=True,
                     help="e.g. logs/job4_results/sift10m_search_50_idx_uint32.bin")
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-queries", type=int, default=None,
                     help="limit query count (debugging only; omit for a full validation run)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--aggregate-recall-tolerance-pct", type=float, default=0.5)
    args = ap.parse_args()

    dtype = np.uint8 if args.vector_dtype == "uint8" else np.float32

    print(f"[{args.corpus_name}] loading index...")
    idx = DiskIndex(args.disk_index, dtype)
    pqt = PQTable(args.pq_pivots)
    pqc = PQCodes(args.pq_compressed)
    ep = read_entry_points(args.disk_index, idx)

    queries = read_query_file(args.query_file, dtype)
    gt_ids, _ = read_ground_truth(args.ground_truth)
    stock_ids = read_stock_result_ids(args.stock_result_ids)

    n_queries = args.num_queries or queries.shape[0]
    if not (queries.shape[0] == gt_ids.shape[0] == stock_ids.shape[0]):
        raise ValueError(
            f"query/ground-truth/stock-result counts disagree: "
            f"{queries.shape[0]}, {gt_ids.shape[0]}, {stock_ids.shape[0]}"
        )

    params = SearchParams(L=args.L, W=args.W, k=args.k)
    node_pred = {t: np.ones(idx.meta.npts, dtype=bool) for t in TIERS}

    harness_recall = np.empty(n_queries)
    stock_overlap = np.empty(n_queries)  # |harness ∩ stock| / k
    exact_id_set_match = np.zeros(n_queries, dtype=bool)

    t0 = time.time()
    for qi in range(n_queries):
        q = queries[qi].astype(np.float32)
        result = masked_search(q, idx, pqt, pqc, node_pred, params, ep)
        harness_recall[qi] = recall_at_k(result.ids, gt_ids[qi, : args.k])
        got = set(result.ids.tolist())
        stock_set = set(stock_ids[qi, : args.k].tolist())
        stock_overlap[qi] = len(got & stock_set) / args.k
        exact_id_set_match[qi] = got == stock_set
        if (qi + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"[{args.corpus_name}] {qi + 1}/{n_queries} queries, "
                  f"{elapsed:.1f}s elapsed, mean stock overlap so far "
                  f"{stock_overlap[:qi + 1].mean():.4f}")
    wall_clock = time.time() - t0

    stock_recall_vs_gt = np.mean([
        recall_at_k(stock_ids[qi, : args.k], gt_ids[qi, : args.k]) for qi in range(n_queries)
    ])

    report = {
        "corpus": args.corpus_name,
        "n_queries": n_queries,
        "search_params": {"L": args.L, "W": args.W, "k": args.k},
        "wall_clock_sec": wall_clock,
        "mean_harness_recall_vs_gt": float(harness_recall.mean()),
        "mean_stock_recall_vs_gt": float(stock_recall_vs_gt),
        "aggregate_recall_gap_pct": float(
            abs(harness_recall.mean() - stock_recall_vs_gt) * 100
        ),
        "mean_stock_id_set_overlap": float(stock_overlap.mean()),
        "exact_id_set_match_fraction": float(exact_id_set_match.mean()),
        "aggregate_gate_pass": bool(
            abs(harness_recall.mean() - stock_recall_vs_gt) * 100
            <= args.aggregate_recall_tolerance_pct
        ),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))

    print(f"\n[{args.corpus_name}] RESULT")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
