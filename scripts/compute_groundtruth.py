#!/usr/bin/env python3
"""Exact k-NN ground truth via batched/blocked GPU distance computation.

Reads base and query vectors in DiskANN's .bin format (uint32 npts, uint32 dim,
then row-major data; dtype given via --dtype), computes exact L2 nearest
neighbors for every query against the *entire* base set on GPU, in blocks
sized to fit in VRAM, and writes results in DiskANN's ground-truth .bin format
(int32 npts, int32 k, then npts*k int32 ids, then npts*k float32 distances).

Before trusting a full-scale run, use --self-test: it generates a small
synthetic base/query set, runs the same blocked-GPU k-NN function (with tiny
block sizes so the block-merge logic is actually exercised) and compares
against a direct numpy brute-force computation over that same small set.
Running brute force over the real (10M-scale) base set on CPU is infeasible
memory-wise, which is why verification uses a small synthetic case rather
than the real corpus.

Usage:
  python compute_groundtruth.py --self-test
  python compute_groundtruth.py --base base.fbin --base-dtype float32 \
      --query query.fbin --query-dtype float32 --k 10 \
      --out gt.bin --query-block 2000 --base-block 200000
"""
import argparse
import struct

import numpy as np
import torch


def read_bin_header(path):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("<ii", f.read(8))
    return npts, dim


def read_bin(path, dtype):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("<ii", f.read(8))
        data = np.fromfile(f, dtype=dtype, count=npts * dim).reshape(npts, dim)
    return data


def write_gt_bin(path, ids, dists):
    npts, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", npts, k))
        ids.astype("<i4").tofile(f)
        dists.astype("<f4").tofile(f)


def brute_force_cpu(query_block, base, k):
    # query_block: (q, d) float64, base: (n, d) float64
    d2 = ((query_block[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
    idx = np.argsort(d2, axis=1)[:, :k]
    dist = np.take_along_axis(d2, idx, axis=1)
    return idx, dist


def blocked_knn_arrays(base_arr, query_arr, k, query_block, base_block, device):
    """Core batched/blocked GPU k-NN over in-memory (or memmapped) arrays.

    base_arr: array-like (n, d), sliceable (memmap or ndarray).
    query_arr: ndarray (q, d) float32.
    Returns (ids int64 (q,k), dists float32 (q,k)).
    """
    base_npts = base_arr.shape[0]
    query_npts = query_arr.shape[0]
    all_ids = np.zeros((query_npts, k), dtype=np.int64)
    all_dists = np.zeros((query_npts, k), dtype=np.float32)

    for qs in range(0, query_npts, query_block):
        qe = min(qs + query_block, query_npts)
        q = torch.from_numpy(np.ascontiguousarray(query_arr[qs:qe])).to(
            device=device, dtype=torch.float32)
        q_sq = (q ** 2).sum(dim=1, keepdim=True)  # (q,1)

        best_dist = None
        best_idx = None
        for bs in range(0, base_npts, base_block):
            be = min(bs + base_block, base_npts)
            b_np = np.ascontiguousarray(base_arr[bs:be]).astype(np.float32)
            b = torch.from_numpy(b_np).to(device=device, dtype=torch.float32)
            b_sq = (b ** 2).sum(dim=1)  # (b,)

            # squared L2: ||q||^2 + ||b||^2 - 2 q.b^T
            d2 = q_sq + b_sq[None, :] - 2.0 * (q @ b.T)
            d2.clamp_(min=0)

            local_k = min(k, d2.shape[1])
            block_dist, block_idx = torch.topk(d2, local_k, dim=1, largest=False, sorted=True)
            block_idx = block_idx + bs

            if best_dist is None:
                best_dist, best_idx = block_dist, block_idx
            else:
                cat_dist = torch.cat([best_dist, block_dist], dim=1)
                cat_idx = torch.cat([best_idx, block_idx], dim=1)
                merged_k = min(k, cat_dist.shape[1])
                best_dist, sel = torch.topk(cat_dist, merged_k, dim=1, largest=False, sorted=True)
                best_idx = torch.gather(cat_idx, 1, sel)

        all_dists[qs:qe] = best_dist.cpu().numpy()
        all_ids[qs:qe] = best_idx.cpu().numpy()
        print(f"queries {qe}/{query_npts}", end="\r")
    print()
    return all_ids, all_dists


def gpu_blocked_knn(base_path, base_dtype, query_path, query_dtype, k,
                     query_block, base_block, device):
    base_npts, base_dim = read_bin_header(base_path)
    query_npts, query_dim = read_bin_header(query_path)
    assert base_dim == query_dim, (base_dim, query_dim)
    print(f"base: {base_npts} x {base_dim} ({base_dtype}); "
          f"query: {query_npts} x {query_dim} ({query_dtype})")

    base_mm = np.memmap(base_path, dtype=base_dtype, mode="r",
                         offset=8, shape=(base_npts, base_dim))
    query_data = read_bin(query_path, query_dtype).astype(np.float32)

    ids, dists = blocked_knn_arrays(base_mm, query_data, k, query_block, base_block, device)
    return ids, dists


def self_test(device):
    """Verify the blocked-GPU k-NN implementation against direct numpy brute
    force on a small synthetic case, using tiny block sizes so the
    cross-block merge logic is actually exercised (not just single-block)."""
    rng = np.random.default_rng(1234)
    n_base, n_query, dim, k = 5000, 50, 32, 10
    base = rng.standard_normal((n_base, dim)).astype(np.float32)
    query = rng.standard_normal((n_query, dim)).astype(np.float32)

    gpu_ids, gpu_dists = blocked_knn_arrays(
        base, query, k, query_block=7, base_block=777, device=device)

    cpu_idx, cpu_dist = brute_force_cpu(query.astype(np.float64), base.astype(np.float64), k)

    mismatches = 0
    for i in range(n_query):
        gpu_set = set(gpu_ids[i].tolist())
        cpu_set = set(cpu_idx[i].tolist())
        if gpu_set != cpu_set:
            if abs(gpu_dists[i].max() - cpu_dist[i].max()) > 1e-3:
                mismatches += 1
                print(f"MISMATCH query {i}: gpu={sorted(gpu_set)} cpu={sorted(cpu_set)}")
    if mismatches:
        raise RuntimeError(
            f"self-test failed: {mismatches}/{n_query} queries mismatched between "
            f"blocked-GPU and numpy brute force. Do not trust the full-scale run.")
    max_abs_err = np.abs(np.sort(gpu_dists, axis=1) - np.sort(cpu_dist, axis=1)).max()
    print(f"self-test passed: {n_query} queries, {n_base} base points, "
          f"blocked-GPU exactly matches numpy brute force (max dist err {max_abs_err:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="Run the small synthetic correctness check and exit.")
    ap.add_argument("--base")
    ap.add_argument("--base-dtype", choices=["float32", "uint8"])
    ap.add_argument("--query")
    ap.add_argument("--query-dtype", choices=["float32", "uint8"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out")
    ap.add_argument("--query-block", type=int, default=2000)
    ap.add_argument("--base-block", type=int, default=200_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if args.self_test:
        self_test(device)
        return

    assert args.base and args.base_dtype and args.query and args.query_dtype and args.out, \
        "--base/--base-dtype/--query/--query-dtype/--out are required unless --self-test"

    dtype_map = {"float32": "<f4", "uint8": "u1"}
    base_dtype = dtype_map[args.base_dtype]
    query_dtype = dtype_map[args.query_dtype]

    ids, dists = gpu_blocked_knn(
        args.base, base_dtype, args.query, query_dtype, args.k,
        args.query_block, args.base_block, device,
    )

    write_gt_bin(args.out, ids, dists)
    print(f"wrote ground truth -> {args.out}")


if __name__ == "__main__":
    main()
