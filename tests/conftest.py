import os
import struct
import subprocess

import numpy as np
import pytest

BUILD_DISK_INDEX_BIN = os.environ.get(
    "DURAVEC_BUILD_DISK_INDEX_BIN",
    "/media/volume/vector/duravec-pilot/diskann_src/build/apps/build_disk_index",
)


def write_fbin_like(path, vectors, dtype):
    n, d = vectors.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        vectors.astype(dtype).tofile(f)


def build_synthetic_disk_index(tmp_path, name, n, dim, dtype, data_type_flag, dist_fn, R, L, B=0.05, M=1, QD=0, seed=0):
    if not os.path.exists(BUILD_DISK_INDEX_BIN):
        pytest.skip(
            f"build_disk_index binary not found at {BUILD_DISK_INDEX_BIN} "
            "(set DURAVEC_BUILD_DISK_INDEX_BIN to override)"
        )
    rng = np.random.default_rng(seed)
    if dtype == np.uint8:
        vectors = rng.integers(0, 256, size=(n, dim), dtype=np.uint8)
    else:
        vectors = rng.standard_normal((n, dim)).astype(dtype)

    data_path = tmp_path / f"{name}_base.bin"
    write_fbin_like(data_path, vectors, dtype)

    prefix = tmp_path / f"{name}_disk_index_{name}_R{R}_L{L}"
    args = [
        BUILD_DISK_INDEX_BIN,
        "--data_type", data_type_flag,
        "--dist_fn", dist_fn,
        "--data_path", str(data_path),
        "--index_path_prefix", str(prefix),
        "-R", str(R), "-L", str(L), "-B", str(B), "-M", str(M), "-T", "4",
    ]
    if QD:
        args += ["--QD", str(QD)]
    subprocess.run(args, check=True, capture_output=True, text=True)

    return {
        "vectors": vectors,
        "data_path": str(data_path),
        "disk_index": f"{prefix}_disk.index",
        "pq_pivots": f"{prefix}_pq_pivots.bin",
        "pq_compressed": f"{prefix}_pq_compressed.bin",
        "n": n,
        "dim": dim,
        "R": R,
        "dtype": dtype,
    }


@pytest.fixture(scope="session")
def synthetic_index_uint8(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("synthetic_uint8")
    return build_synthetic_disk_index(
        tmp_path, "u8idx", n=500, dim=32, dtype=np.uint8,
        data_type_flag="uint8", dist_fn="l2", R=16, L=32, QD=8,
    )


@pytest.fixture(scope="session")
def synthetic_index_float32(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("synthetic_float32")
    return build_synthetic_disk_index(
        tmp_path, "f32idx", n=600, dim=24, dtype=np.float32,
        data_type_flag="float", dist_fn="l2", R=20, L=40, QD=8,
    )
