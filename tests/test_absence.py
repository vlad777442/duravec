import struct

import numpy as np
import pytest

from harness.absence import read_ground_truth, absence_ceiling, recall_at_k


def test_read_ground_truth_roundtrip(tmp_path):
    ids = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="<i4")
    dists = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype="<f4")
    path = tmp_path / "toy.gt.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", 2, 4))
        ids.tofile(f)
        dists.tofile(f)

    got_ids, got_dists = read_ground_truth(str(path))
    assert np.array_equal(got_ids, ids)
    assert np.allclose(got_dists, dists)


def test_absence_ceiling_exact_fraction():
    true_topk = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ])
    codes_ok = np.zeros(8, dtype=bool)
    codes_ok[[0, 1, 4]] = True  # query0: 2/4 present, query1: 1/4 present

    ceiling = absence_ceiling(true_topk, codes_ok)
    assert ceiling.tolist() == pytest.approx([0.5, 0.25])


def test_absence_ceiling_all_present_or_absent():
    true_topk = np.array([[0, 1, 2]])
    assert absence_ceiling(true_topk, np.ones(3, dtype=bool)).tolist() == pytest.approx([1.0])
    assert absence_ceiling(true_topk, np.zeros(3, dtype=bool)).tolist() == pytest.approx([0.0])


def test_recall_at_k_full_and_partial_overlap():
    true_topk = np.array([1, 2, 3, 4, 5])
    assert recall_at_k(np.array([1, 2, 3, 4, 5]), true_topk) == pytest.approx(1.0)
    assert recall_at_k(np.array([1, 2, 999, 998, 997]), true_topk) == pytest.approx(2 / 5)
    assert recall_at_k(np.array([100, 200]), true_topk) == pytest.approx(0.0)


def test_recall_at_k_ignores_order_and_duplicates_do_not_inflate():
    true_topk = np.array([1, 2, 3])
    # a result set can't have duplicates in practice, but the function
    # should not double count if it somehow did
    assert recall_at_k(np.array([1, 1, 2]), true_topk) == pytest.approx(2 / 3)
