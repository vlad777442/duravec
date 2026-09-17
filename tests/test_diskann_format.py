import numpy as np
import pytest

from harness.diskann_format import DiskIndex, PQTable, PQCodes, read_bin_header, SECTOR_LEN


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_disk_index_matches_ground_truth_vectors(fixture_name, request):
    idx_info = request.getfixturevalue(fixture_name)
    idx = DiskIndex(idx_info["disk_index"], idx_info["dtype"])

    assert idx.meta.npts == idx_info["n"]
    assert idx.meta.ndims == idx_info["dim"]
    assert idx.R == idx_info["R"]
    assert idx.meta.nnodes_per_sector > 0

    for node_id in [0, 1, idx_info["n"] // 2, idx_info["n"] - 1]:
        vec = idx.get_vector(node_id)
        assert np.array_equal(vec, idx_info["vectors"][node_id])


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_disk_index_adjacency_is_well_formed(fixture_name, request):
    idx_info = request.getfixturevalue(fixture_name)
    idx = DiskIndex(idx_info["disk_index"], idx_info["dtype"])
    n = idx_info["n"]

    degrees = []
    for node_id in range(n):
        deg = idx.get_degree(node_id)
        nbrs = idx.get_neighbors(node_id)
        assert len(nbrs) == deg
        assert deg <= idx.R
        assert deg > 0
        assert np.all(nbrs < n)
        assert np.all(nbrs >= 0)
        assert node_id not in nbrs  # no self-loops
        degrees.append(deg)


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_disk_index_file_size_matches_sector_arithmetic(fixture_name, request):
    import os
    idx_info = request.getfixturevalue(fixture_name)
    idx = DiskIndex(idx_info["disk_index"], idx_info["dtype"])

    n_sectors = -(-idx.meta.npts // idx.meta.nnodes_per_sector)  # ceil
    expected_size = (n_sectors + 1) * SECTOR_LEN
    assert expected_size == idx.meta.disk_index_file_size
    assert expected_size == os.path.getsize(idx_info["disk_index"])


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_medoid_is_valid_node_with_neighbors(fixture_name, request):
    idx_info = request.getfixturevalue(fixture_name)
    idx = DiskIndex(idx_info["disk_index"], idx_info["dtype"])
    medoid = idx.meta.medoid
    assert 0 <= medoid < idx_info["n"]
    assert idx.get_degree(medoid) > 0


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_pq_codes_and_pivots_are_consistent_with_vectors(fixture_name, request):
    idx_info = request.getfixturevalue(fixture_name)
    pqt = PQTable(idx_info["pq_pivots"])
    pqc = PQCodes(idx_info["pq_compressed"])

    assert pqc.npts == idx_info["n"]
    assert pqt.ndims == idx_info["dim"]

    node_ids = np.array([0, 1, idx_info["n"] // 2, idx_info["n"] - 1])
    codes = pqc.get_batch(node_ids)
    approx = pqt.decode(codes)
    true_vecs = idx_info["vectors"][node_ids].astype(np.float32)

    err = np.sqrt(((approx - true_vecs) ** 2).sum(axis=1))
    norm = np.sqrt((true_vecs ** 2).sum(axis=1))
    # lossy quantization: nonzero but bounded relative error, never exact
    assert np.all(err > 0)
    assert np.all(err / np.maximum(norm, 1e-6) < 2.0)


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_adc_distance_table_ranks_consistently_with_exact_distance(fixture_name, request):
    """The PQ ADC distance (used for CODES-only scoring under masking) should
    correlate with true L2 distance well enough to be a sane ranking proxy —
    not exact, since it's lossy, but not scrambled either."""
    idx_info = request.getfixturevalue(fixture_name)
    pqt = PQTable(idx_info["pq_pivots"])
    pqc = PQCodes(idx_info["pq_compressed"])

    query = idx_info["vectors"][0].astype(np.float32)
    table = pqt.adc_distance_table(query)

    n = idx_info["n"]
    all_codes = pqc.get_batch(np.arange(n))
    adc_dist = table[np.arange(pqt.n_chunks)[None, :], all_codes].sum(axis=1)

    true_vecs = idx_info["vectors"].astype(np.float32)
    exact_dist = ((true_vecs - query) ** 2).sum(axis=1)

    exact_rank = np.argsort(exact_dist)[:20]
    adc_rank = np.argsort(adc_dist)[:20]
    overlap = len(set(exact_rank.tolist()) & set(adc_rank.tolist()))
    assert overlap >= 10  # loose correlation check, not exact-match


@pytest.mark.parametrize("fixture_name", ["synthetic_index_uint8", "synthetic_index_float32"])
def test_get_neighbors_batch_matches_per_node_accessors(fixture_name, request):
    idx_info = request.getfixturevalue(fixture_name)
    idx = DiskIndex(idx_info["disk_index"], idx_info["dtype"])
    n = idx_info["n"]

    rng = np.random.default_rng(7)
    sample = rng.choice(n, size=min(50, n), replace=False)
    degrees, neighbor_matrix = idx.get_neighbors_batch(sample)

    for row, node_id in enumerate(sample):
        expected = idx.get_neighbors(int(node_id))
        deg = degrees[row]
        assert deg == len(expected)
        assert np.array_equal(neighbor_matrix[row, :deg], expected)


def test_read_bin_header_matches_numpy_fromfile(tmp_path):
    import struct
    path = tmp_path / "toy.bin"
    data = np.arange(20, dtype="<f4").reshape(5, 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", 5, 4))
        data.tofile(f)
    nrows, ncols = read_bin_header(str(path))
    assert (nrows, ncols) == (5, 4)
