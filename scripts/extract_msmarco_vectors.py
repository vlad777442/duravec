#!/usr/bin/env python3
"""Extract raw float32 vectors from a prebuilt Pyserini FAISS flat index and
write them in DiskANN's .fbin format (uint32 npts, uint32 dim, then row-major
float32 data), plus a held-out query set sampled from the same corpus.

Usage:
  python extract_msmarco_vectors.py \
      --index-dir /path/to/extracted/faiss-flat.msmarco-v1-passage.bge-base-en-v1.5.20240107 \
      --out-base /path/to/corpora/msmarco/base.8.84M.fbin \
      --out-query /path/to/corpora/msmarco/query.10K.fbin \
      --num-queries 10000 \
      --seed 42
"""
import argparse
import os
import struct

import faiss
import numpy as np


def find_index_file(index_dir):
    for name in os.listdir(index_dir):
        path = os.path.join(index_dir, name)
        if os.path.isfile(path) and name in ("index", "faiss.index"):
            return path
    # fall back: single largest file in the directory
    candidates = [os.path.join(index_dir, n) for n in os.listdir(index_dir)]
    candidates = [c for c in candidates if os.path.isfile(c)]
    return max(candidates, key=os.path.getsize)


def write_fbin(path, vectors):
    n, d = vectors.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        vectors.astype("<f4").tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--out-query", required=True)
    ap.add_argument("--num-queries", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=200_000)
    args = ap.parse_args()

    index_path = find_index_file(args.index_dir)
    print(f"loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    n = index.ntotal
    d = index.d
    print(f"index reports ntotal={n} dim={d}")

    rng = np.random.default_rng(args.seed)
    query_ids = rng.choice(n, size=args.num_queries, replace=False)
    query_ids.sort()
    query_id_set = np.zeros(n, dtype=bool)
    query_id_set[query_ids] = True

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_query), exist_ok=True)

    n_base = n - args.num_queries
    base_out = np.lib.format.open_memmap(
        args.out_base + ".tmp.npy", mode="w+", dtype="<f4", shape=(n_base, d)
    )
    query_out = np.empty((args.num_queries, d), dtype="<f4")

    base_write_pos = 0
    query_write_pos = 0
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        block = index.reconstruct_n(start, end - start)
        mask = query_id_set[start:end]
        n_q_in_block = int(mask.sum())
        if n_q_in_block:
            query_out[query_write_pos:query_write_pos + n_q_in_block] = block[mask]
            query_write_pos += n_q_in_block
        n_b_in_block = (end - start) - n_q_in_block
        if n_b_in_block:
            base_out[base_write_pos:base_write_pos + n_b_in_block] = block[~mask]
            base_write_pos += n_b_in_block
        print(f"reconstructed {end}/{n}", end="\r")

    print()
    assert base_write_pos == n_base, (base_write_pos, n_base)
    assert query_write_pos == args.num_queries, (query_write_pos, args.num_queries)

    print(f"writing base vectors -> {args.out_base} ({n_base} x {d})")
    write_fbin(args.out_base, base_out)
    del base_out
    os.remove(args.out_base + ".tmp.npy")

    print(f"writing query vectors -> {args.out_query} ({args.num_queries} x {d})")
    write_fbin(args.out_query, query_out)

    print("done")


if __name__ == "__main__":
    main()
