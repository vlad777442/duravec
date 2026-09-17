import numpy as np
import pytest

from harness.diskann_format import DiskIndex
from harness.mask import (
    make_rank_grouping, compute_group_byte_stats, mask_from_fraction,
    mask_from_groups, mask_bytes, TIERS,
)


@pytest.fixture
def idx_and_grouping(synthetic_index_uint8):
    info = synthetic_index_uint8
    idx = DiskIndex(info["disk_index"], info["dtype"])
    rng = np.random.default_rng(1)
    order = rng.permutation(info["n"])
    grouping = make_rank_grouping("random_seed1", order, group_size=17, order_description="random")
    return info, idx, grouping


def test_grouping_rejects_non_permutation():
    with pytest.raises(ValueError):
        make_rank_grouping("bad", np.array([0, 0, 1]), group_size=1, order_description="x")


def test_grouping_covers_every_node_exactly_once(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    n = info["n"]
    assert grouping.group_of_node.shape == (n,)
    assert grouping.group_of_node.min() == 0
    assert grouping.group_of_node.max() == grouping.n_groups - 1
    # every group actually has members
    counts = np.bincount(grouping.group_of_node, minlength=grouping.n_groups)
    assert np.all(counts > 0)
    assert counts.sum() == n


def test_fraction_zero_and_one_restore_nothing_and_everything(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    n = info["n"]

    m0 = mask_from_fraction(grouping, 0.0)
    preds0 = m0.node_predicates(grouping)
    for t in TIERS:
        assert not preds0[t].any()

    m1 = mask_from_fraction(grouping, 1.0)
    preds1 = m1.node_predicates(grouping)
    for t in TIERS:
        assert preds1[t].all()


def test_fraction_sweep_is_monotonic_and_cumulative(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    prev_restored = None
    for frac in [0.1, 0.3, 0.5, 0.8, 1.0]:
        m = mask_from_fraction(grouping, frac)
        preds = m.node_predicates(grouping)["CODES"]
        n_restored = preds.sum()
        if prev_restored is not None:
            assert n_restored >= prev_restored
            # cumulative: everything restored at a lower fraction stays restored
            prev_preds, _ = prev_restored_arr
            assert np.all(prev_preds <= preds)
        prev_restored = n_restored
        prev_restored_arr = (preds.copy(), n_restored)


def test_byte_stats_actual_is_less_than_or_equal_reserved(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    stats = compute_group_byte_stats(grouping, idx)
    assert np.all(stats.adj_bytes_actual <= stats.adj_bytes_reserved)
    # R=16 in this synthetic fixture and every node saturates to max degree
    # (see test_diskann_format), so actual should equal reserved here
    assert np.allclose(stats.adj_bytes_actual, stats.adj_bytes_reserved)


def test_mask_bytes_matches_direct_sum(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    stats = compute_group_byte_stats(grouping, idx)

    groups = {"CODES": [0, 1, 2], "ADJ": [0, 1], "VEC": [2]}
    m = mask_from_groups(grouping, groups)
    b = mask_bytes(m, stats, adj_convention="actual")

    assert b["CODES"] == pytest.approx(stats.codes_bytes[[0, 1, 2]].sum())
    assert b["ADJ"] == pytest.approx(stats.adj_bytes_actual[[0, 1]].sum())
    assert b["VEC"] == pytest.approx(stats.vec_bytes[[2]].sum())
    assert b["total"] == pytest.approx(b["CODES"] + b["ADJ"] + b["VEC"])
    assert b["ADJ_other_convention"] == pytest.approx(stats.adj_bytes_reserved[[0, 1]].sum())


def test_mask_bytes_never_double_counts_across_disjoint_groups(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    stats = compute_group_byte_stats(grouping, idx)
    full = mask_from_fraction(grouping, 1.0)
    b_full = mask_bytes(full, stats)
    total_actual_bytes = stats.codes_bytes.sum() + stats.adj_bytes_actual.sum() + stats.vec_bytes.sum()
    assert b_full["total"] == pytest.approx(total_actual_bytes)


def test_node_predicates_rejects_mismatched_grouping(idx_and_grouping):
    info, idx, grouping = idx_and_grouping
    other = make_rank_grouping("other", np.arange(info["n"]), group_size=5, order_description="sequential")
    m = mask_from_fraction(grouping, 0.5)
    with pytest.raises(ValueError):
        m.node_predicates(other)
