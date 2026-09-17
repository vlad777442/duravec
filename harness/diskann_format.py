"""Readers for DiskANN's on-disk formats: the generic .bin container, the
sector-packed disk.index (full-precision vector + adjacency per node), and
the multi-section pq_pivots.bin / pq_compressed.bin codes files.

Layout facts here are taken directly from diskann_src (cpp_main, pinned
commit in pilot/provenance.md), not inferred:
  - generic .bin: int32 nrows, int32 ncols, then nrows*ncols elements
    (src/utils.h: save_bin / load_bin).
  - disk.index sector 0: a .bin-format record of uint64 metadata fields
    (src/disk_utils.cpp create_disk_layout, output_file_meta).
  - disk.index node layout: [full-precision coords][uint32 degree]
    [degree * uint32 neighbor ids], zero-padded to max_node_len, several
    nodes packed per 4096B sector (same function).
  - pq_pivots.bin: an outer .bin header of size_t byte-offsets pointing at
    three inner .bin sections (pivot tables, centroid, chunk_offsets)
    (src/pq.cpp FixedChunkPQTable::load_pq_centroid_bin).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SECTOR_LEN = 4096
NUM_PQ_CENTROIDS = 256


def read_bin_header(path, offset=0):
    """Return (nrows, ncols) from a generic DiskANN .bin file/section."""
    with open(path, "rb") as f:
        f.seek(offset)
        nrows, ncols = struct.unpack("<ii", f.read(8))
    return nrows, ncols


def load_bin_section(path, dtype, offset=0):
    """Read one generic .bin section (header + row-major data) at a byte
    offset within `path`. Returns an (nrows, ncols) ndarray of `dtype`."""
    nrows, ncols = read_bin_header(path, offset)
    dtype = np.dtype(dtype)
    count = nrows * ncols
    data = np.fromfile(path, dtype=dtype, count=count, offset=offset + 8)
    if data.size != count:
        raise ValueError(
            f"{path}: expected {count} {dtype} elements at offset {offset}, got {data.size}"
        )
    return data.reshape(nrows, ncols)


@dataclass(frozen=True)
class DiskIndexMeta:
    npts: int
    ndims: int
    medoid: int
    max_node_len: int
    nnodes_per_sector: int
    vamana_frozen_num: int
    vamana_frozen_loc: int
    append_reorder_data: bool
    disk_index_file_size: int


def read_disk_index_meta(disk_index_path) -> DiskIndexMeta:
    nrows, ncols = read_bin_header(disk_index_path, offset=0)
    if ncols != 1:
        raise ValueError(f"{disk_index_path}: unexpected metadata ncols={ncols}")
    vals = np.fromfile(disk_index_path, dtype="<u8", count=nrows, offset=8)
    if nrows == 9:
        (
            npts, ndims, medoid, max_node_len, nnodes_per_sector,
            vamana_frozen_num, vamana_frozen_loc, append_reorder_data,
            disk_index_file_size,
        ) = vals.tolist()
    elif nrows == 12:
        raise NotImplementedError(
            "disk.index was built with --use_reorder_data (append_reorder_data=1); "
            "this parser only supports the format actually used for the pilot indexes "
            "(no reorder-data section). See pilot/provenance.md correction entry."
        )
    else:
        raise ValueError(f"{disk_index_path}: unexpected metadata row count {nrows}")
    return DiskIndexMeta(
        npts=npts, ndims=ndims, medoid=medoid, max_node_len=max_node_len,
        nnodes_per_sector=nnodes_per_sector, vamana_frozen_num=vamana_frozen_num,
        vamana_frozen_loc=vamana_frozen_loc, append_reorder_data=bool(append_reorder_data),
        disk_index_file_size=disk_index_file_size,
    )


class DiskIndex:
    """Memory-mapped reader over a stock DiskANN disk.index file.

    `vector_dtype` must match the corpus dtype the index was built from
    (uint8 for SIFT, float32 for Deep10M/MS MARCO) — it is not recoverable
    from the file itself, since the on-disk element width alone can't
    distinguish e.g. two different float layouts.
    """

    def __init__(self, disk_index_path, vector_dtype):
        self.path = disk_index_path
        self.meta = read_disk_index_meta(disk_index_path)
        self.vector_dtype = np.dtype(vector_dtype)

        vec_bytes = self.meta.ndims * self.vector_dtype.itemsize
        remaining = self.meta.max_node_len - vec_bytes - 4
        if remaining < 0 or remaining % 4 != 0:
            raise ValueError(
                f"{disk_index_path}: max_node_len={self.meta.max_node_len} does not "
                f"decompose into ndims*itemsize({vec_bytes}) + 4 + R*4 for dtype "
                f"{self.vector_dtype}; wrong vector_dtype for this index?"
            )
        self.R = remaining // 4

        if self.meta.nnodes_per_sector == 0:
            raise NotImplementedError(
                "multi-sector-per-node layout (max_node_len > SECTOR_LEN) not needed "
                "for these three corpora and not implemented"
            )

        self._mmap = np.memmap(disk_index_path, dtype=np.uint8, mode="r")
        self._vec_bytes = vec_bytes

    def _node_offset(self, node_id: int) -> int:
        if not (0 <= node_id < self.meta.npts):
            raise IndexError(node_id)
        sector = node_id // self.meta.nnodes_per_sector
        slot = node_id % self.meta.nnodes_per_sector
        return (sector + 1) * SECTOR_LEN + slot * self.meta.max_node_len

    def get_vector(self, node_id: int) -> np.ndarray:
        off = self._node_offset(node_id)
        raw = self._mmap[off: off + self._vec_bytes]
        return raw.view(self.vector_dtype).copy()

    def get_degree(self, node_id: int) -> int:
        off = self._node_offset(node_id) + self._vec_bytes
        return int(self._mmap[off: off + 4].view("<u4")[0])

    def get_neighbors(self, node_id: int) -> np.ndarray:
        off = self._node_offset(node_id) + self._vec_bytes
        degree = int(self._mmap[off: off + 4].view("<u4")[0])
        nbr_off = off + 4
        return self._mmap[nbr_off: nbr_off + degree * 4].view("<u4").copy()

    def _node_view(self) -> np.ndarray:
        """(npts_padded, max_node_len) view over the data sectors, one row per
        node slot in on-disk order (no copy). npts_padded may exceed npts by
        the last sector's unused slots; callers slice to [:npts]."""
        data = self._mmap[SECTOR_LEN:]
        n_sectors = data.shape[0] // SECTOR_LEN
        sectors = data[: n_sectors * SECTOR_LEN].reshape(n_sectors, SECTOR_LEN)
        per_sector = sectors[:, : self.meta.nnodes_per_sector * self.meta.max_node_len]
        nodes = per_sector.reshape(n_sectors * self.meta.nnodes_per_sector, self.meta.max_node_len)
        return nodes[: self.meta.npts]

    def get_neighbors_batch(self, node_ids: np.ndarray):
        """Vectorized neighbor fetch for many nodes at once: returns
        (degrees[m], neighbor_matrix[m, R]) where neighbor_matrix's columns
        past each row's degree are padding (not meaningful -- the reserved
        slot beyond actual degree, not necessarily zero-safe to treat as
        "no neighbor"; callers must mask by degree, not by value). Used for
        full-graph passes (in-degree, BFS, traversal replay) where a
        per-node Python loop over up to 10M nodes would dominate runtime."""
        node_ids = np.asarray(node_ids, dtype=np.int64)
        nodes = self._node_view()[node_ids]  # (m, max_node_len), gather view
        degrees = nodes[:, self._vec_bytes: self._vec_bytes + 4].view("<u4").reshape(-1).copy()
        nbr_block = nodes[:, self._vec_bytes + 4: self._vec_bytes + 4 + self.R * 4]
        neighbor_matrix = nbr_block.view("<u4").copy()
        return degrees, neighbor_matrix

    def get_all_degrees(self) -> np.ndarray:
        """Vectorized degree extraction for all nodes — no per-node Python
        loop, needed since importance proxies and the byte model touch every
        node in a 10M-node graph."""
        nodes = self._node_view()
        degree_col = nodes[:, self._vec_bytes: self._vec_bytes + 4]
        return degree_col.view("<u4").reshape(-1).copy()


class PQTable:
    """Decoded pq_pivots.bin: per-chunk centroid tables + dim-mean centroid,
    enough to reconstruct approximate vectors or build ADC distance tables."""

    def __init__(self, pq_pivots_path):
        offsets = load_bin_section(pq_pivots_path, "<u8", offset=0).reshape(-1)
        nr = offsets.shape[0]
        if nr not in (4, 5):
            raise ValueError(f"{pq_pivots_path}: expected 4 or 5 offsets, got {nr}")
        use_old_filetype = nr == 5

        self.tables = load_bin_section(pq_pivots_path, "<f4", offset=int(offsets[0]))
        if self.tables.shape[0] != NUM_PQ_CENTROIDS:
            raise ValueError(
                f"{pq_pivots_path}: expected {NUM_PQ_CENTROIDS} centroids, "
                f"got {self.tables.shape[0]}"
            )
        self.ndims = self.tables.shape[1]

        centroid = load_bin_section(pq_pivots_path, "<f4", offset=int(offsets[1]))
        if centroid.shape != (self.ndims, 1):
            raise ValueError(f"{pq_pivots_path}: unexpected centroid shape {centroid.shape}")
        self.centroid = centroid.reshape(-1)

        chunk_offsets_idx = 3 if use_old_filetype else 2
        chunk_offsets = load_bin_section(
            pq_pivots_path, "<u4", offset=int(offsets[chunk_offsets_idx])
        )
        self.chunk_offsets = chunk_offsets.reshape(-1)
        self.n_chunks = self.chunk_offsets.shape[0] - 1

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes: (n, n_chunks) uint8 -> (n, ndims) float32 approx vectors."""
        out = np.empty((codes.shape[0], self.ndims), dtype="<f4")
        for c in range(self.n_chunks):
            lo, hi = int(self.chunk_offsets[c]), int(self.chunk_offsets[c + 1])
            out[:, lo:hi] = self.tables[codes[:, c], lo:hi] + self.centroid[lo:hi]
        return out

    def adc_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Squared-L2 asymmetric distance table: (n_chunks, 256) — table[c, k]
        is the squared distance from `query`'s chunk-c slice to centroid k's
        chunk-c slice. Sum table[c, codes[:, c]] over c for per-point ADC
        distance, matching what search over a CODES-only node uses."""
        table = np.empty((self.n_chunks, NUM_PQ_CENTROIDS), dtype="<f4")
        q = query.astype("<f4") - self.centroid
        for c in range(self.n_chunks):
            lo, hi = int(self.chunk_offsets[c]), int(self.chunk_offsets[c + 1])
            diff = self.tables[:, lo:hi] - q[lo:hi]
            table[c] = np.sum(diff * diff, axis=1)
        return table


@dataclass(frozen=True)
class EntryPoints:
    """Candidate search entry points, with their real (full-precision)
    coordinate vectors. Single-shot builds (SIFT10M, Deep10M) have exactly
    one, taken from the disk.index metadata sector. Sharded/merged builds
    (MS MARCO) can have several -- one per shard -- in a sibling
    `<prefix>_medoids.bin` / `<prefix>_centroids.bin` pair, and stock search
    picks whichever is exactly closest to the query
    (pq_flash_index.cpp: `use_medoids_data_as_centroids`,
    `_dist_cmp_float->compare(query_float, _centroid_data + ...)`)."""

    node_ids: np.ndarray   # int, (m,)
    vectors: np.ndarray    # float32, (m, ndims) -- real coords, not PQ


def read_entry_points(disk_index_path: str, disk_index: DiskIndex) -> EntryPoints:
    import os

    medoids_path = disk_index_path + "_medoids.bin"
    centroids_path = disk_index_path + "_centroids.bin"
    if os.path.exists(medoids_path) and os.path.exists(centroids_path):
        medoid_ids = load_bin_section(medoids_path, "<u4").reshape(-1)
        centroids = load_bin_section(centroids_path, "<f4")
        if centroids.shape[0] != medoid_ids.shape[0]:
            raise ValueError(
                f"{medoids_path}/{centroids_path}: medoid count mismatch "
                f"({medoid_ids.shape[0]} vs {centroids.shape[0]})"
            )
        return EntryPoints(node_ids=medoid_ids.astype(np.int64), vectors=centroids.astype("<f4"))
    medoid = disk_index.meta.medoid
    vec = disk_index.get_vector(medoid).astype("<f4")
    return EntryPoints(node_ids=np.array([medoid], dtype=np.int64), vectors=vec.reshape(1, -1))


class PQCodes:
    """Memory-mapped reader over pq_compressed.bin (per-point PQ codes)."""

    def __init__(self, pq_compressed_path):
        npts, n_chunks = read_bin_header(pq_compressed_path)
        self.npts = npts
        self.n_chunks = n_chunks
        self._mmap = np.memmap(
            pq_compressed_path, dtype="<u1", mode="r", offset=8,
            shape=(npts, n_chunks),
        )

    def get(self, node_id: int) -> np.ndarray:
        return np.asarray(self._mmap[node_id])

    def get_batch(self, node_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._mmap[node_ids])
