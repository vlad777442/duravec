import time

import numpy as np
import pytest

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_entry_points
from harness.mask import make_rank_grouping, compute_group_byte_stats, mask_from_fraction
from harness.runner import RunSpec, run_masked_sweep_point, load_sweep_records, run_id


def brute_force_topk_batch(vectors, queries, k):
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        d = ((vectors.astype(np.float32) - q.astype(np.float32)) ** 2).sum(axis=1)
        out[i] = np.argsort(d)[:k]
    return out


@pytest.fixture
def scenario(synthetic_index_uint8):
    info = synthetic_index_uint8
    idx = DiskIndex(info["disk_index"], info["dtype"])
    pqt = PQTable(info["pq_pivots"])
    pqc = PQCodes(info["pq_compressed"])
    ep = read_entry_points(info["disk_index"], idx)

    rng = np.random.default_rng(11)
    order = rng.permutation(info["n"])
    grouping = make_rank_grouping("random_seed11", order, group_size=25, order_description="random")
    byte_stats = compute_group_byte_stats(grouping, idx)

    query_idx = rng.choice(info["n"], size=8, replace=False)
    queries = info["vectors"][query_idx]
    true_topk = brute_force_topk_batch(info["vectors"], queries, k=10)

    return dict(info=info, idx=idx, pqt=pqt, pqc=pqc, ep=ep, grouping=grouping,
                byte_stats=byte_stats, queries=queries, true_topk=true_topk)


def test_run_produces_well_formed_record(scenario, tmp_path):
    s = scenario
    mask = mask_from_fraction(s["grouping"], 0.5)
    spec = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups={t: sorted(mask.restored_groups[t]) for t in mask.restored_groups},
        search_params=dict(L=32, W=4, k=10, io_limit=None),
        label="test frac=0.5",
    )
    record = run_masked_sweep_point(
        spec, str(tmp_path), mask, s["grouping"], s["byte_stats"], s["idx"], s["pqt"], s["pqc"],
        s["ep"], s["queries"], s["true_topk"],
    )
    assert record["n_queries"] == 8
    assert 0.0 <= record["mean_recall"] <= 1.0
    assert 0.0 <= record["mean_absence_ceiling"] <= 1.0
    assert len(record["per_query"]["recall"]) == 8
    assert record["bytes"]["total"] > 0
    assert record["spec"]["label"] == "test frac=0.5"


def test_run_is_resumable_without_recomputation(scenario, tmp_path):
    s = scenario
    mask = mask_from_fraction(s["grouping"], 0.3)
    spec = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups={t: sorted(mask.restored_groups[t]) for t in mask.restored_groups},
        search_params=dict(L=32, W=4, k=10, io_limit=None),
    )
    r1 = run_masked_sweep_point(
        spec, str(tmp_path), mask, s["grouping"], s["byte_stats"], s["idx"], s["pqt"], s["pqc"],
        s["ep"], s["queries"], s["true_topk"],
    )
    t0 = time.time()
    r2 = run_masked_sweep_point(
        spec, str(tmp_path), mask, s["grouping"], s["byte_stats"], s["idx"], s["pqt"], s["pqc"],
        s["ep"], s["queries"], s["true_topk"],
    )
    elapsed = time.time() - t0
    assert r1 == r2
    assert elapsed < 0.05  # loaded from disk, not recomputed


def test_label_does_not_affect_run_id(scenario):
    s = scenario
    mask = mask_from_fraction(s["grouping"], 0.4)
    restored = {t: sorted(mask.restored_groups[t]) for t in mask.restored_groups}
    spec_a = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups=restored, search_params=dict(L=32, W=4, k=10, io_limit=None),
        label="measurement2 wording",
    )
    spec_b = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups=restored, search_params=dict(L=32, W=4, k=10, io_limit=None),
        label="measurement3 wording, totally different text",
    )
    assert run_id(spec_a) == run_id(spec_b)


def test_different_masks_produce_different_run_ids_and_are_both_persisted(scenario, tmp_path):
    s = scenario
    mask_a = mask_from_fraction(s["grouping"], 0.2)
    mask_b = mask_from_fraction(s["grouping"], 0.9)
    spec_a = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups={t: sorted(mask_a.restored_groups[t]) for t in mask_a.restored_groups},
        search_params=dict(L=32, W=4, k=10, io_limit=None),
    )
    spec_b = RunSpec(
        corpus="synthetic", grouping_name=s["grouping"].name,
        order_description=s["grouping"].order_description,
        restored_groups={t: sorted(mask_b.restored_groups[t]) for t in mask_b.restored_groups},
        search_params=dict(L=32, W=4, k=10, io_limit=None),
    )
    assert run_id(spec_a) != run_id(spec_b)

    run_masked_sweep_point(spec_a, str(tmp_path), mask_a, s["grouping"], s["byte_stats"],
                            s["idx"], s["pqt"], s["pqc"], s["ep"], s["queries"], s["true_topk"])
    run_masked_sweep_point(spec_b, str(tmp_path), mask_b, s["grouping"], s["byte_stats"],
                            s["idx"], s["pqt"], s["pqc"], s["ep"], s["queries"], s["true_topk"])

    records = load_sweep_records(str(tmp_path))
    assert len(records) == 2
    bytes_totals = sorted(r["bytes"]["total"] for r in records)
    assert bytes_totals[0] < bytes_totals[1]  # 0.2 restores fewer bytes than 0.9
