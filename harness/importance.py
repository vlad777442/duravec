"""Importance proxies (HARNESS_JOBS.md 6): in-degree, hop-distance from the
medoid, and traversal frequency. Each is computed once per index and meant
to be cached to disk by the caller (np.save) -- these are the expensive,
full-graph passes.

Full-graph adjacency passes are batched through DiskIndex.get_neighbors_batch
rather than looped node-by-node in Python: a 10M-node graph with mean degree
~60-100 means a naive per-node Python loop is impractical (confirmed: a
vectorized full-graph neighbor fetch over all of SIFT10M takes ~7s; a
Python-level per-node loop over the same data is roughly two orders of
magnitude slower).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .diskann_format import DiskIndex
from .search import masked_search, SearchParams
from .diskann_format import PQTable, PQCodes, EntryPoints
from .mask import TIERS


def _valid_neighbor_mask(degrees: np.ndarray, R: int) -> np.ndarray:
    col_idx = np.arange(R)[None, :]
    return col_idx < degrees[:, None]


def compute_in_degree(disk_index: DiskIndex, batch_size: int = 200_000) -> np.ndarray:
    """One pass over adjacency (HARNESS_JOBS.md 6): in_degree[n] = number of
    stored edges pointing at n, counted from the actual per-node degree
    (not the reserved R slot)."""
    npts = disk_index.meta.npts
    in_degree = np.zeros(npts, dtype=np.int64)
    for start in range(0, npts, batch_size):
        end = min(start + batch_size, npts)
        ids = np.arange(start, end, dtype=np.int64)
        degrees, nbrs = disk_index.get_neighbors_batch(ids)
        valid = _valid_neighbor_mask(degrees, disk_index.R)
        flat = nbrs[valid]
        in_degree += np.bincount(flat, minlength=npts)
    return in_degree


def compute_hop_distance(disk_index: DiskIndex, source, batch_size: int = 500_000) -> np.ndarray:
    """One BFS from `source` (HARNESS_JOBS.md 6), frontier-batched rather
    than node-by-node. `source` may be a single node id or several (e.g.
    MS MARCO's multiple candidate medoids, see diskann_format.EntryPoints)
    -- multi-source BFS gives hop-distance-to-nearest-entry-point, the
    natural generalization of "hop distance from the medoid" once there is
    more than one, and is what the CODES-fallback-seed distance in
    search.py needs regardless of which entry point a given query selects.
    Unreached nodes (shouldn't happen on a connected Vamana graph, but the
    graph is directed via ADJ so isolated-in-degree nodes are possible) get
    max_reached_distance + 1, so they sort last in any hop-distance-based
    restore order rather than crashing the caller."""
    npts = disk_index.meta.npts
    sources = np.atleast_1d(np.asarray(source, dtype=np.int64))
    dist = np.full(npts, -1, dtype=np.int64)
    dist[sources] = 0
    frontier = np.unique(sources)
    d = 0
    while frontier.size > 0:
        d += 1
        pieces = []
        for start in range(0, frontier.size, batch_size):
            chunk = frontier[start: start + batch_size]
            degrees, nbrs = disk_index.get_neighbors_batch(chunk)
            valid = _valid_neighbor_mask(degrees, disk_index.R)
            pieces.append(nbrs[valid])
        candidates = np.unique(np.concatenate(pieces)) if pieces else np.array([], dtype=np.int64)
        unreached = candidates[dist[candidates] == -1]
        dist[unreached] = d
        frontier = unreached

    still_unreached = dist == -1
    if still_unreached.any():
        reached_max = dist[~still_unreached].max() if (~still_unreached).any() else 0
        dist[still_unreached] = reached_max + 1
    return dist


def split_workload_eval_queries(n_queries: int, seed: int = 1234, workload_fraction: float = 0.5):
    """Split a corpus's held-out query set into a traversal-frequency
    workload and an evaluation set, disjoint, per pilot decision
    2026-09-16: 5K/5K (not a smaller workload slice), because with only ~2K
    queries most of a 10M-node graph gets zero visits and whatever breaks
    that enormous tie class becomes the real ordering -- 5K/5K shrinks the
    tie class and the caller (compute_traversal_frequency) breaks remaining
    ties randomly with a recorded seed, reporting the never-visited
    fraction so the amount of real signal is auditable."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_queries)
    n_workload = int(round(n_queries * workload_fraction))
    return perm[:n_workload], perm[n_workload:]


def compute_traversal_frequency(
    disk_index: DiskIndex,
    pq_table: PQTable,
    pq_codes: PQCodes,
    entry_points: EntryPoints,
    workload_queries: np.ndarray,   # (m, ndims), the workload subset's vectors
    search_params: SearchParams,
    tie_break_seed: int,
):
    """Replay the workload query set over the *intact, unmasked* graph and
    count node visits (= expansions) -- this is what a frequency-based
    warm-up policy (B3) would use, so it must be genuinely measured, not
    approximated (HARNESS_JOBS.md 6).

    Returns (visit_count[npts] int64, never_visited_fraction float,
    tie_break_key[npts] float) -- tie_break_key is a fixed random
    permutation (seeded) for the caller to use as the secondary sort key so
    the enormous zero-visit tie class doesn't silently order by whatever
    the caller's array happens to be sorted by otherwise (e.g. node id,
    which would smuggle in a structural bias).
    """
    npts = disk_index.meta.npts
    visit_count = np.zeros(npts, dtype=np.int64)
    node_pred = {t: np.ones(npts, dtype=bool) for t in TIERS}

    for q in workload_queries:
        result = masked_search(q.astype(np.float32), disk_index, pq_table, pq_codes, node_pred, search_params, entry_points)
        visit_count[result.expanded_ids] += 1

    never_visited_fraction = float((visit_count == 0).mean())
    tie_break_key = np.random.default_rng(tie_break_seed).random(npts)
    return visit_count, never_visited_fraction, tie_break_key


def order_by_score(score: np.ndarray, descending: bool = True, tie_break_key: Optional[np.ndarray] = None) -> np.ndarray:
    """Turn a per-node importance score into a restore-priority order
    (highest importance restored first, by default), with an explicit,
    auditable tie-break instead of an implicit stable-sort-by-node-id one."""
    if tie_break_key is None:
        tie_break_key = np.arange(score.shape[0])
    primary = -score if descending else score
    return np.lexsort((tie_break_key, primary))
