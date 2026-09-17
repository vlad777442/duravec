"""Ground-truth reading and the exact absence ceiling (HARNESS_JOBS.md 5.3).

absence_ceiling(q, mask) = |{n in true_topk(q) : n has CODES under mask}| / k

Everything between measured recall and this ceiling is broken navigation:
neighbors present but unreachable. Computed exactly from ground truth, never
inferred by differencing two runs.
"""
from __future__ import annotations

import struct

import numpy as np


def read_ground_truth(gt_path: str):
    """DiskANN GT format (scripts/compute_groundtruth.py write_gt_bin):
    int32 npts, int32 k, then npts*k int32 ids, then npts*k float32 dists."""
    with open(gt_path, "rb") as f:
        npts, k = struct.unpack("<ii", f.read(8))
    ids = np.fromfile(gt_path, dtype="<i4", count=npts * k, offset=8).reshape(npts, k)
    dists = np.fromfile(
        gt_path, dtype="<f4", count=npts * k, offset=8 + npts * k * 4
    ).reshape(npts, k)
    return ids, dists


def absence_ceiling(true_topk_ids: np.ndarray, codes_ok: np.ndarray) -> np.ndarray:
    """true_topk_ids: (n_queries, k) int ground-truth neighbor ids.
    codes_ok: bool[npts], CODES-tier restored predicate under some mask.
    Returns per-query ceiling, (n_queries,) float in [0, 1]."""
    has_codes = codes_ok[true_topk_ids]
    return has_codes.mean(axis=1)


def recall_at_k(result_ids: np.ndarray, true_topk_ids: np.ndarray) -> float:
    """Single-query recall@k: |result ∩ true_topk| / k. Ties/order don't
    matter, only set membership -- matches DiskANN's own recall definition."""
    k = true_topk_ids.shape[0]
    return len(set(result_ids.tolist()) & set(true_topk_ids.tolist())) / k
