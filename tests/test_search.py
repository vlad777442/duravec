import collections

import numpy as np
import pytest

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points
from harness.mask import Mask, TIERS
from harness.search import masked_search, SearchParams


def bfs_hop_distance(disk_index, source):
    dist = np.full(disk_index.meta.npts, -1, dtype=np.int64)
    dist[source] = 0
    q = collections.deque([source])
    while q:
        u = q.popleft()
        for v in disk_index.get_neighbors(u).tolist():
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    dist[dist == -1] = dist.max() + 1  # unreached: treat as maximally far
    return dist


@pytest.fixture
def loaded(synthetic_index_uint8):
    info = synthetic_index_uint8
    idx = DiskIndex(info["disk_index"], info["dtype"])
    pqt = PQTable(info["pq_pivots"])
    pqc = PQCodes(info["pq_compressed"])
    ep = read_entry_points(info["disk_index"], idx)
    return info, idx, pqt, pqc, ep


def full_predicates(n):
    ones = np.ones(n, dtype=bool)
    return {t: ones for t in TIERS}


def brute_force_topk(vectors, query, k):
    d = ((vectors.astype(np.float32) - query.astype(np.float32)) ** 2).sum(axis=1)
    return np.argsort(d)[:k]


def test_unmasked_search_finds_true_nearest_neighbors(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    preds = full_predicates(n)
    params = SearchParams(L=64, W=8, k=10)

    rng = np.random.default_rng(2)
    overlaps = []
    for _ in range(20):
        qi = rng.integers(0, n)
        query = info["vectors"][qi].astype(np.float32)
        result = masked_search(query, idx, pqt, pqc, preds, params, ep)
        assert result.miss_count == 0
        assert result.code_only_count == 0
        assert not result.used_fallback_seed

        true_topk = set(brute_force_topk(info["vectors"], query, 10).tolist())
        got = set(result.ids.tolist())
        overlaps.append(len(true_topk & got) / 10)

    # exact full-precision node distances are used for every expanded node
    # here (nothing masked), so with generous L this should track brute
    # force closely, not just loosely.
    assert np.mean(overlaps) > 0.9


def test_codes_masking_excludes_nodes_from_candidacy_and_results(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    rng = np.random.default_rng(3)
    excluded = set(rng.choice(n, size=n // 2, replace=False).tolist())
    codes_ok = np.array([i not in excluded for i in range(n)], dtype=bool)
    preds = {"CODES": codes_ok, "ADJ": np.ones(n, dtype=bool), "VEC": np.ones(n, dtype=bool)}

    params = SearchParams(L=64, W=8, k=10)
    query = info["vectors"][0].astype(np.float32)

    hop_dist = bfs_hop_distance(idx, ep.node_ids[0])

    result = masked_search(query, idx, pqt, pqc, preds, params, ep, hop_distance_from_entry=hop_dist)
    assert not any(i in excluded for i in result.ids.tolist())


def test_adj_masking_produces_misses_without_blocking(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    # mask ADJ off for every node except a handful near the entry point, so
    # the search is forced to hit misses but must still terminate cleanly.
    adj_ok = np.zeros(n, dtype=bool)
    adj_ok[int(ep.node_ids[0])] = True
    for nb in idx.get_neighbors(int(ep.node_ids[0]))[:5]:
        adj_ok[nb] = True
    preds = {"CODES": np.ones(n, dtype=bool), "ADJ": adj_ok, "VEC": np.ones(n, dtype=bool)}

    params = SearchParams(L=64, W=8, k=10)
    query = info["vectors"][0].astype(np.float32)
    result = masked_search(query, idx, pqt, pqc, preds, params, ep)

    assert result.miss_count > 0
    assert result.num_expanded > 0  # search still made progress, didn't block


def test_vec_masking_forces_code_only_results(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    vec_ok = np.zeros(n, dtype=bool)  # nothing has VEC
    preds = {"CODES": np.ones(n, dtype=bool), "ADJ": np.ones(n, dtype=bool), "VEC": vec_ok}

    params = SearchParams(L=64, W=8, k=10)
    query = info["vectors"][0].astype(np.float32)
    result = masked_search(query, idx, pqt, pqc, preds, params, ep)

    assert result.code_only_count == result.num_expanded
    assert result.code_only_count > 0


def test_entry_point_fallback_seed_when_entry_lacks_codes(loaded):
    info, idx, pqt, pqc, ep = loaded
    n = info["n"]
    entry_id = int(ep.node_ids[0])
    codes_ok = np.ones(n, dtype=bool)
    codes_ok[entry_id] = False
    preds = {"CODES": codes_ok, "ADJ": np.ones(n, dtype=bool), "VEC": np.ones(n, dtype=bool)}

    hop_dist = bfs_hop_distance(idx, entry_id)
    params = SearchParams(L=64, W=8, k=10)
    query = info["vectors"][0].astype(np.float32)
    result = masked_search(query, idx, pqt, pqc, preds, params, ep, hop_distance_from_entry=hop_dist)

    assert result.used_fallback_seed
    assert entry_id not in result.ids.tolist()
