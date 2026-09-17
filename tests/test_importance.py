import collections

import numpy as np
import pytest

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points
from harness.importance import (
    compute_in_degree, compute_hop_distance, split_workload_eval_queries,
    compute_traversal_frequency, order_by_score,
)
from harness.search import SearchParams


def brute_force_in_degree(idx, n):
    counts = np.zeros(n, dtype=np.int64)
    for node_id in range(n):
        for nb in idx.get_neighbors(node_id):
            counts[nb] += 1
    return counts


def brute_force_hop_distance(idx, n, source):
    dist = np.full(n, -1, dtype=np.int64)
    dist[source] = 0
    q = collections.deque([source])
    while q:
        u = q.popleft()
        for v in idx.get_neighbors(u).tolist():
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    unreached = dist == -1
    if unreached.any():
        dist[unreached] = dist[~unreached].max() + 1
    return dist


@pytest.fixture
def loaded(synthetic_index_uint8):
    info = synthetic_index_uint8
    idx = DiskIndex(info["disk_index"], info["dtype"])
    pqt = PQTable(info["pq_pivots"])
    pqc = PQCodes(info["pq_compressed"])
    ep = read_entry_points(info["disk_index"], idx)
    return info, idx, pqt, pqc, ep


def test_in_degree_matches_brute_force(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    expected = brute_force_in_degree(idx, n)
    got = compute_in_degree(idx, batch_size=97)  # deliberately not a clean divisor of n
    assert np.array_equal(expected, got)


def test_hop_distance_matches_brute_force_bfs(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    source = int(ep.node_ids[0])
    expected = brute_force_hop_distance(idx, n, source)
    got = compute_hop_distance(idx, source, batch_size=53)
    assert np.array_equal(expected, got)
    assert got[source] == 0


def test_hop_distance_multi_source_is_min_over_single_source_bfs(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    rng = np.random.default_rng(42)
    sources = rng.choice(n, size=3, replace=False)

    per_source = [brute_force_hop_distance(idx, n, int(s)) for s in sources]
    expected = np.minimum.reduce(per_source)

    got = compute_hop_distance(idx, sources, batch_size=53)
    assert np.array_equal(expected, got)
    assert np.all(got[sources] == 0)


def test_split_workload_eval_is_disjoint_and_covers_all_queries():
    workload, eval_ids = split_workload_eval_queries(10000, seed=1234, workload_fraction=0.5)
    assert len(workload) == 5000
    assert len(eval_ids) == 5000
    assert set(workload.tolist()).isdisjoint(set(eval_ids.tolist()))
    assert set(workload.tolist()) | set(eval_ids.tolist()) == set(range(10000))


def test_traversal_frequency_visits_are_plausible(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    rng = np.random.default_rng(5)
    workload = info["vectors"][rng.choice(n, size=30, replace=False)].astype(np.float32)
    params = SearchParams(L=32, W=4, k=10)

    visit_count, never_visited_frac, tie_key = compute_traversal_frequency(
        idx, pqt, pqc, ep, workload, params, tie_break_seed=99,
    )
    assert visit_count.shape == (n,)
    assert visit_count.sum() > 0
    assert 0.0 <= never_visited_frac <= 1.0
    assert tie_key.shape == (n,)
    # entry point should be visited on essentially every query (it's always
    # the seed, or the fallback-seed target)
    assert visit_count[int(ep.node_ids[0])] > 0


def test_order_by_score_descending_with_tie_break():
    score = np.array([1.0, 5.0, 5.0, 2.0])
    tie_key = np.array([0.9, 0.1, 0.2, 0.5])
    order = order_by_score(score, descending=True, tie_break_key=tie_key)
    # nodes 1 and 2 tie at score=5; tie_key breaks in favor of node 1 (lower key)
    assert order.tolist() == [1, 2, 3, 0]


def test_order_by_score_default_tie_break_is_deterministic_by_node_id():
    score = np.array([3.0, 3.0, 1.0])
    order = order_by_score(score, descending=True)
    assert order.tolist() == [0, 1, 2]
