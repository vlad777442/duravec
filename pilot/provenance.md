# DuraVec pilot — provenance log

Append-only. Each entry records what was done, when, and how to reproduce it. Do not
reconstruct this file from shell history after the fact — write it as each step completes.

## Environment

- Single node, 16 vCPU, 58GB RAM, 1x NVIDIA A100 (sliced instance, 20GB VRAM), driver CUDA 13.0,
  no CUDA toolkit (`nvcc`) installed.
- Persistent storage: `/media/volume/vector` (196GB ext4 volume, `UUID=5fa8b392-d5d7-4171-9ec5-fd1a0820a1a7`),
  added to `/etc/fstab` on 2026-09-16 so it survives reboots. All pilot data lives under
  `/media/volume/vector/duravec-pilot/`. The repo itself stays on the (smaller, GitHub-backed)
  instance root disk.
- Python venv: `/media/volume/vector/duravec-pilot/venv` (faiss-cpu, numpy).

## DiskANN

- Repository: https://github.com/microsoft/DiskANN
- **Branch note:** `main` has been rewritten in Rust (weekly releases, v0.59.0 as of
  2026-09-11). The classic C++ implementation — `build_disk_index`, the R/L/PQ-byte CLI, and
  the on-disk codes/adjacency/full-precision tier layout that PREREGISTRATION.md's H1
  byte-accounting assumes — lives on the separate **`cpp_main`** branch.
- Pinned commit: `78256bbab4685e1774e78d331e081a153be26823` (cpp_main HEAD as of 2026-09-16;
  last commit to that branch was 2026-05-05, so it predates the Rust migration and is stable).
- Cloned to `/media/volume/vector/duravec-pilot/diskann_src`, checked out detached at the
  pinned commit, submodules initialized.
- Build deps installed via apt: `make cmake g++ libaio-dev libgoogle-perftools-dev
  clang-format libboost-all-dev libmkl-full-dev` (Ubuntu 24.04 packages; MKL 2020.4.304-4).
  Note: `libmkl-rt` postinst tries to show an interactive whiptail license dialog even under
  `apt-get install -y`; use `DEBIAN_FRONTEND=noninteractive` or it hangs.
- Built with `cmake -DCMAKE_BUILD_TYPE=Release .. && make -j16` in
  `diskann_src/build`.

## Build parameters (Job 2)

Sourced from the NeurIPS'21 Big ANN Benchmarks repo
(`harsha-simhadri/big-ann-benchmarks`, `algos-2021.yaml`), which defines a `diskann-t2`
baseline for `bigann-10M` (SIFT10M) and `deep-10M` at exactly this scale:

- **SIFT10M and Deep10M:** R=100, L=100, search_DRAM_budget (B)=0.3 GB, build_DRAM_budget (M)=15 GB.
  `--PQ_disk_bytes` left at its default (0), which auto-derives the on-disk PQ byte budget from B.
- **MS MARCO (BGE-base-en-v1.5, 768-dim float):** not part of that benchmark; R=100/L=100 kept
  for graph-quality consistency, but B/M must be recomputed for the much larger per-vector byte
  footprint (768×4 bytes vs. SIFT's 128×1 or Deep's 96×4) before building — not yet done as of
  this entry.

## Corpora

### SIFT10M (bigann-10M)

- Source: `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin`,
  first 10M points obtained via HTTP Range request (bytes 0-1280000007) rather than downloading
  the full 1B-point (128GB) file.
- Downloaded 2026-09-16 to
  `/media/volume/vector/duravec-pilot/corpora/sift10m/base.10M.u8bin.crop`
  (1,280,000,008 bytes, matches expected size exactly).
- Header patched in place (the crop retains the original file's `npts=1000000000` header) to
  `npts=10000000, dim=128`, then renamed to `base.10M.u8bin`.
- Query set: full file `query.public.10K.u8bin` from the same base_url (10,000 queries, 128-d
  uint8) — downloaded whole since it's already small.
- Published GT reference downloaded for cross-check only (not used as the pilot's own ground
  truth, per Job 3's requirement to compute it ourselves):
  `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M` ->
  `bigann-10M.gt_reference`.

### Deep10M

- Source: `https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin`,
  first 10M points obtained via HTTP Range request (bytes 0-3840000007).
- Downloaded 2026-09-16 to
  `/media/volume/vector/duravec-pilot/corpora/deep10m/base.10M.fbin.crop`
  (3,840,000,008 bytes, matches expected size exactly).
- Header patched from `npts=1000000000, dim=96` to `npts=10000000, dim=96`, renamed to
  `base.10M.fbin`.
- Query set: full file `query.public.10K.fbin` from the same base_url (10,000 queries, 96-d
  float32).
- Published GT reference downloaded for cross-check only:
  `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M` ->
  `deep-10M.gt_reference`.

### MS MARCO passages

- Embedding model: **BGE-base-en-v1.5** (BAAI, 768-dim, float32). Chosen over TCT-ColBERT-v2 as
  the more contemporary general-purpose default in current vector-index benchmark literature;
  confirmed with the PI before downloading.
- Source: prebuilt FAISS flat (exact) index from the Pyserini/Anserini project (Univ. of
  Waterloo), not generated — avoids spending A100 time on embedding 8.84M passages.
  - URL: `https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.msmarco-v1-passage.bge-base-en-v1.5.20240107.tar.gz`
  - MD5 verified: `b21fb6abee3be6da3b6f39c9f6d9f280` (matches Pyserini's published metadata).
  - 8,841,823 passages (matches the standard MS MARCO passage corpus count).
- Downloaded and MD5-verified 2026-09-16 to
  `/media/volume/vector/duravec-pilot/embeddings/`.
- Query set: **no natural MS MARCO dev-query embeddings were used.** Per-PI decision, a random
  subset of corpus passages is held out as queries (same convention as the SIFT/Deep query
  sets), rather than encoding the real ~6980 MS MARCO dev queries. This means MS MARCO queries
  are in-corpus, unlike SIFT/Deep/real-IR usage — noted here so it isn't lost track of later.
- Extraction to raw `.fbin` (DiskANN format) via `scripts/extract_msmarco_vectors.py`
  (`faiss.IndexFlat.reconstruct_n` + held-out query sampling) — in progress as of this entry.

## Index builds (Job 2 — completed 2026-09-16)

All three built with `diskann_src/build/apps/build_disk_index` (pinned commit above), via
`scripts/build_diskann_index.sh`. Full stdout logs preserved at
`logs/job2_build_{sift10m,deep10m,msmarco}.log`.

| Corpus | Command params | Wall time | Disk index size | Notes |
|---|---|---|---|---|
| SIFT10M | `-R 100 -L 100 -B 0.3 -M 15 -T 16` | 130m47s | 5,851,435,008 B | Single-shot build (fit in 7.13GB RAM estimate). Auto-derived PQ = 32 bytes/vector (confirmed in log: "Compressing 128-dimensional data into 32 bytes per vector"). |
| Deep10M | `-R 100 -L 100 -B 0.3 -M 15 -T 16` | 146m42s | 8,192,004,096 B | Single-shot build (9.75GB RAM estimate). Auto-derived PQ = 32 bytes/vector. |
| MS MARCO | `-R 100 -L 100 -B 1.0 -M 32 -T 16 --QD 32` | 118m42s (2nd attempt; see incident below) | 36,175,151,104 B | Sharded build (3 shards, replication factor 2, ~17.66M augmented points) since raw data (27GB) exceeds the 32GB single-shard heuristic threshold. Explicit `--QD 32` pins PQ to 32 bytes/vector, matching the other two corpora. |

**Incident:** the first MS MARCO build attempt was launched concurrently with the SIFT10M and
Deep10M builds (all three requesting `-T 16` on a 16-core machine). It was OOM-killed by the
kernel (confirmed via `dmesg`: `Killed process ... build_disk_inde ... anon-rss:47754484kB`)
while building its second shard (6.98M points), because its own ~47.7GB peak RSS plus the other
two builds' concurrent usage (~18GB) exceeded the machine's 58GB RAM. Fix: waited for SIFT10M
and Deep10M to finish, deleted MS MARCO's stale partial temp files, and reran MS MARCO alone
with identical parameters — succeeded. Lesson for later stages: DiskANN's own shard-size RAM
estimate undershot actual peak usage by roughly 2x for this 768-dim corpus; do not run multiple
`build_disk_index` invocations concurrently on this machine regardless of their individual `-M`
budgets.

## Ground truth (Job 3 — completed 2026-09-16)

Exact k=10 L2 nearest neighbors computed via `scripts/compute_groundtruth.py`: a batched/blocked
GPU implementation (squared-L2 via `||q||^2+||b||^2-2q.b^T`, blocked over both queries and base
to fit VRAM, `torch.topk` per block merged across blocks — not a naive per-query loop).

- **Correctness verification (before trusting the full run):** `--self-test` generates a small
  synthetic case (5000 base points, 50 queries, dim 32) and runs the *same* blocked-GPU function
  with deliberately tiny block sizes (query_block=7, base_block=777, forcing >1 block on both
  axes so the cross-block merge logic is actually exercised), then compares against direct numpy
  brute force over the same data. Passed: exact top-k set match, max distance error 9.4e-06
  (expected float32-vs-float64 precision gap). Running brute force over the real 10M/8.8M-scale
  base sets on CPU is infeasible memory-wise (a full float64 pairwise distance matrix against the
  real base would need tens of GB to terabytes), which is why verification uses a small
  synthetic case rather than a subset of the real corpus.
- Command: `--query-block 2000 --base-block 250000` for all three corpora (keeps the per-block
  distance matrix at ~2GB, well within the 20GB VRAM budget).
- Output: `ground_truth/{sift10m,deep10m,msmarco}.gt.bin` (DiskANN GT format: int32 npts, int32
  k, then npts×k int32 ids, then npts×k float32 distances). Each computed in well under a minute
  (GPU compute is fast; disk I/O for memmapped base reads dominates).
- **Cross-check against the published reference GT** (downloaded in Job 2 for exactly this
  purpose, not used as the pilot's own ground truth): SIFT10M matches
  `bigann-10M.gt_reference` at 100% top-1 exact-match rate and 99.99% mean top-10 set overlap;
  Deep10M matches `deep-10M.gt_reference` at 99.94% / 99.96%. Residual gap consistent with
  floating-point tie-breaking on near-equal distances, not a correctness bug. MS MARCO has no
  published reference to cross-check against, since its query set is our own held-out corpus
  subset rather than the official MS MARCO dev queries (see Corpora section above).

## Baseline characterization (Job 4 — completed 2026-09-16)

**Search parameters ("stock"):** `-L 50 -W 4(beamwidth) -T 16 --num_nodes_to_cache 0`, i.e.
the exact `query-args` from the same published NeurIPS'21 big-ann-benchmarks `diskann-t2`
config used for the SIFT10M/Deep10M build params, applied to MS MARCO too for consistency
(no published query-args exist for MS MARCO since it isn't part of that benchmark).

**Source patch:** DiskANN's stock `search_disk_index` only reports mean and P99.9 latency, but
PREREGISTRATION.md's H2 threshold (`TTRS(0.9, 2×P99)`) is defined against **P99** specifically.
Added a P99 column using the same existing `get_percentile_stats` helper the binary already
calls for P99.9 (two-line change: `apps/search_disk_index.cpp`, computing
`get_percentile_stats<float>(stats, query_num, 0.99, ...)` alongside the existing 0.999 call and
printing it). Rebuilt `search_disk_index` only. This is the first departure from the pinned
`cpp_main` commit's stock code; the diff is small, additive, and does not change any existing
measurement, only exposes one already-computed statistic at a different percentile.

| Corpus | Recall@10 | Mean latency | P99 latency | P99.9 latency | QPS |
|---|---|---|---|---|---|
| SIFT10M | 98.01% | 31.90ms | **104.5ms** | 1001.1ms | 500.05 |
| Deep10M | 95.54% | 36.18ms | **248.5ms** | 921.6ms | 441.05 |
| MS MARCO | 83.53% | 41.43ms | **772.1ms** | 934.4ms | 385.17 |

**Flagging per the job's instructions** ("compare to published numbers... flag any discrepancy
loudly"): SIFT10M and Deep10M recall@10 (98.01%, 95.54%) are in the range expected from
published DiskANN R/L sweeps at this scale and are not treated as anomalous. **MS MARCO's much
lower recall (83.53%) is a real, reportable effect, not a build error**: forcing the same 32
bytes/vector PQ budget (`--QD 32`, chosen deliberately in Job 2 for cross-corpus byte-budget
comparability) is a far more aggressive compression ratio at 768 dimensions (32/768 = 4.2%) than
at SIFT's 128 (25%) or Deep's 96 (33%), which plausibly explains the gap. No published DiskANN
recall number exists for this exact BGE-base-en-v1.5+R100/L100+QD32 combination to compare
against, so this isn't a disagreement-with-literature case requiring a stop — it's a
characteristic of the design choice worth carrying into the write-up. Not re-tuned without
discussing first, per the job's instructions.

**Index size on disk, by tier:**

| Corpus | disk.index (adjacency + PQ-compressed vectors, sector-packed) | pq_pivots + pq_compressed (codes tier, loaded fully into RAM) | Total served index |
|---|---|---|---|
| SIFT10M | 5,851,435,008 B | 320,135,844 B | 6,171,570,852 B |
| Deep10M | 8,192,004,096 B | 320,102,948 B | 8,512,107,044 B |
| MS MARCO | 36,175,160,348 B (incl. small centroids/medoids files) | 283,412,100 B | 36,458,563,204 B |

**Important caveat for future tier-separation work (out of scope here, noted for the ablation
harness spec):** stock DiskANN's on-disk layout bundles each node's PQ-compressed vector
together with its adjacency (neighbor list) in the same sector of `disk.index` — they are not
independently restorable tiers in this format. Full-precision vectors are **not stored on disk
at all** by default; they exist only in the original corpus files and transient build-time
structures, and would require the `--use_reorder_data`/`append_reorder_data` build option to be
appended as a separate on-disk block. PREREGISTRATION.md's H1 counts "codes, adjacency, and
full-precision vectors" as separately restored tiers, so the ablation harness building on these
baseline indexes will need either that reorder-data option or a custom disk layout — the
indexes built here are stock baselines for recall/latency/size characterization, not yet in a
tier-separable form.

`search_disk_index` result files (returned ids/dists per query) saved under
`logs/job4_results/` for reference; full stdout in `logs/job4_{sift10m,deep10m,msmarco}.log`.

## Status as of this entry (2026-09-16)

All four PILOT_JOBS.md jobs complete: persistent storage, DiskANN pin + build (with one small,
documented patch), SIFT10M/Deep10M/MS MARCO corpora and queries prepared, all three disk indexes
built, ground truth computed and cross-validated, baseline recall/latency/size characterized and
flagged where relevant. Data-prep scope for this spec is done; the ablation harness, masking
logic, restoration curves, and Ceph-touching work are explicitly out of scope here per the job
file and belong to a separate spec.
