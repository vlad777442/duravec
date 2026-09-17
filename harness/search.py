"""Masked beam search, mirroring DiskANN's cached_beam_search
(src/pq_flash_index.cpp, pinned cpp_main commit) closely enough that with
nothing masked it must reproduce stock recall (HARNESS_JOBS.md measurement
1).

Traced from source, not reimplemented from memory:
  - The candidate list ("retset") that drives traversal order is always
    ranked by PQ/ADC distance, for every node, visited or not -- even the
    entry-point medoid's very first insertion uses the ADC estimate, not an
    exact distance. Exact full-precision distance is only ever computed for
    nodes that get expanded (sector-read), and only feeds the *final*
    result ranking (`full_retset`), never the traversal queue.
  - `beam_width` (W) batches how many closest-unexpanded candidates are
    expanded per hop. An earlier version of this comment claimed W has no
    effect on recall, only on hop count/wall-clock -- that's wrong, and
    measurement 5 (harness/../pilot/provenance.md, "B5 beam-width sweep")
    caught it empirically: recall rises measurably with W at fixed L,
    because the candidate list's capacity-bounded eviction means a larger
    W expands several close-but-unreconsidered candidates off one shared
    snapshot before their discoveries feed back in, which is the classical
    wider-beam-explores-more-broadly effect. L still has a 2-5x larger
    effect than W in every case measured, but both are genuine
    search-effort knobs, not just L.
  - Final top-k is the k smallest-exact-distance nodes among those actually
    expanded during the search, not among all discovered/visited nodes.

Masking semantics (HARNESS_JOBS.md 4):
  - No CODES on a candidate: it cannot be scored at all, so it is never
    inserted into the candidate list -- filtered at the moment its parent
    tries to add it, exactly like DiskANN's own dummy-point / filter skips.
  - No ADJ on an expanded node: it still gets exact-scored (if VEC present)
    and can appear in results, but contributes zero neighbors to explore.
    Counted as a miss; policy is always-skip (never retried).
  - No VEC on an expanded node: it keeps its ADC distance as its final
    ranking score instead of an exact one ("code-only" result). Counted
    per query.
  - Entry point selection: sharded/merged builds (MS MARCO) can have
    several candidate medoids (see diskann_format.EntryPoints); stock
    picks whichever is exactly closest to the query by real-vector L2, not
    by PQ/ADC distance -- reproduced here the same way.
  - Medoid seeding under masking: if the exact-distance-selected entry
    point lacks CODES under a mask, fall back to the CODES-having node
    closest (by hop distance from that same entry point, computed once,
    unmasked) to it -- the most defensible reading of "highest-ranked"
    available without inventing a new proxy.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diskann_format import DiskIndex, PQTable, PQCodes, EntryPoints


@dataclass(frozen=True)
class SearchParams:
    L: int  # search list capacity
    W: int  # beam width: candidates expanded per hop (I/O batching only)
    k: int = 10
    io_limit: Optional[int] = None  # None = unlimited, matches stock default


@dataclass
class SearchResult:
    ids: np.ndarray          # up to k node ids, sorted by final ranking distance
    dists: np.ndarray        # matching distances (exact where available, ADC else)
    hops: int
    num_expanded: int
    miss_count: int          # expanded nodes with no ADJ
    code_only_count: int     # expanded nodes with no VEC (result kept at ADC dist)
    used_fallback_seed: bool
    expanded_ids: np.ndarray  # every node id actually expanded this query,
                               # unsorted; the traversal-frequency proxy's
                               # "node visits" and B5 IO-cost accounting
                               # both read this rather than re-deriving it.


class _CandidateList:
    """Bounded sorted list matching DiskANN's NeighborPriorityQueue: capacity
    L, sorted by distance, tracks which entries have been expanded."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._ids: list[int] = []
        self._dists: list[float] = []
        self._expanded: list[bool] = []

    def insert(self, node_id: int, dist: float) -> None:
        if len(self._ids) >= self.capacity and dist >= self._dists[-1]:
            return
        pos = bisect.bisect_left(self._dists, dist)
        self._ids.insert(pos, node_id)
        self._dists.insert(pos, dist)
        self._expanded.insert(pos, False)
        if len(self._ids) > self.capacity:
            del self._ids[-1]
            del self._dists[-1]
            del self._expanded[-1]

    def has_unexpanded(self) -> bool:
        return any(not e for e in self._expanded)

    def closest_unexpanded(self) -> int:
        for i, e in enumerate(self._expanded):
            if not e:
                self._expanded[i] = True
                return self._ids[i]
        raise RuntimeError("no unexpanded candidate")


def masked_search(
    query: np.ndarray,
    disk_index: DiskIndex,
    pq_table: PQTable,
    pq_codes: PQCodes,
    node_pred: dict,           # tier -> bool[npts], from Mask.node_predicates
    params: SearchParams,
    entry_points: EntryPoints,
    hop_distance_from_entry: Optional[np.ndarray] = None,  # for fallback seeding
) -> SearchResult:
    codes_ok, adj_ok, vec_ok = node_pred["CODES"], node_pred["ADJ"], node_pred["VEC"]
    query = query.astype("<f4")

    # exact-distance entry-point selection, matching stock search exactly
    # (real vectors, not PQ/ADC) -- see EntryPoints docstring.
    diffs = entry_points.vectors - query[None, :]
    exact_dists = np.sum(diffs * diffs, axis=1)
    entry = int(entry_points.node_ids[np.argmin(exact_dists)])

    used_fallback_seed = False
    if not codes_ok[entry]:
        used_fallback_seed = True
        if hop_distance_from_entry is None:
            raise ValueError(
                "selected entry point lacks CODES under this mask and no "
                "hop_distance_from_entry was supplied for fallback seeding"
            )
        candidates = np.where(codes_ok)[0]
        if candidates.size == 0:
            return SearchResult(
                ids=np.array([], dtype=np.int64), dists=np.array([], dtype="<f4"),
                hops=0, num_expanded=0, miss_count=0, code_only_count=0,
                used_fallback_seed=True, expanded_ids=np.array([], dtype=np.int64),
            )
        best = candidates[np.argmin(hop_distance_from_entry[candidates])]
        seed = int(best)
    else:
        seed = int(entry)

    adc_table = pq_table.adc_distance_table(query)

    def adc_dist(node_ids: np.ndarray) -> np.ndarray:
        codes = pq_codes.get_batch(node_ids)
        return adc_table[np.arange(pq_table.n_chunks)[None, :], codes].sum(axis=1)

    seed_dist = float(adc_dist(np.array([seed]))[0])

    retset = _CandidateList(params.L)
    visited = {seed}
    retset.insert(seed, seed_dist)

    full_retset: list[tuple[int, float, bool]] = []  # (id, dist, is_exact)
    hops = 0
    num_ios = 0
    miss_count = 0
    code_only_count = 0
    io_limit = params.io_limit if params.io_limit is not None else float("inf")

    while retset.has_unexpanded() and num_ios < io_limit:
        frontier = []
        num_seen = 0
        while retset.has_unexpanded() and len(frontier) < params.W and num_seen < params.W:
            frontier.append(retset.closest_unexpanded())
            num_seen += 1
        if not frontier:
            break
        hops += 1

        for node_id in frontier:
            num_ios += 1
            has_vec = bool(vec_ok[node_id])
            if has_vec:
                vec = disk_index.get_vector(node_id).astype("<f4")
                dist = float(np.sum((vec - query) ** 2))
                is_exact = True
            else:
                dist = float(adc_dist(np.array([node_id]))[0])
                is_exact = False
                code_only_count += 1
            full_retset.append((node_id, dist, is_exact))

            if not adj_ok[node_id]:
                miss_count += 1
                continue

            nbrs = disk_index.get_neighbors(node_id)
            nbrs = nbrs[codes_ok[nbrs]]  # CODES-absent neighbors: dropped at expansion
            new_nbrs = [n for n in nbrs.tolist() if n not in visited]
            if not new_nbrs:
                continue
            for n in new_nbrs:
                visited.add(n)
            new_nbrs_arr = np.array(new_nbrs, dtype=np.int64)
            dists = adc_dist(new_nbrs_arr)
            for n, d in zip(new_nbrs, dists):
                retset.insert(int(n), float(d))

    expanded_ids = np.array([t[0] for t in full_retset], dtype=np.int64)
    full_retset.sort(key=lambda t: t[1])
    top = full_retset[: params.k]
    ids = np.array([t[0] for t in top], dtype=np.int64)
    dists = np.array([t[1] for t in top], dtype="<f4")

    return SearchResult(
        ids=ids, dists=dists, hops=hops, num_expanded=len(full_retset),
        miss_count=miss_count, code_only_count=code_only_count,
        used_fallback_seed=used_fallback_seed, expanded_ids=expanded_ids,
    )
