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
- Python venv: `/media/volume/vector/duravec-pilot/venv` (faiss-cpu, numpy, pytest, matplotlib).
- Harness sweep records follow the same convention: the bulky per-run JSON provenance records
  from measurements 2 and 3 (~100MB, 591 files) live at
  `/media/volume/vector/duravec-pilot/harness-results/{measurement2,measurement3}/`, with
  `pilot/results/measurement{2,3}` in the repo as symlinks to them — added 2026-09-17 after they
  were initially written directly into the repo, which doesn't belong on the small GitHub-backed
  root disk per this same convention. Measurements 4 and 5's records are small enough (a few MB)
  to live directly in the repo under `pilot/results/`.

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

## Correction to Job 4's disk.index tier caveat (2026-09-16)

**The "Important caveat for future tier-separation work" note above (Job 4, baseline
characterization) is wrong.** It claims full-precision vectors are not stored in `disk.index`
by default and that `--use_reorder_data` would be needed. Re-derived from `disk_utils.cpp`
(`create_disk_layout`, pinned `cpp_main` commit) and confirmed empirically, not just by
arithmetic:

- Per-node layout in `disk.index` is `[full-precision coords][uint32 degree][degree * uint32
  neighbor ids]`, zero-padded to `max_node_len = ndims*itemsize + 4 + R*4`, several nodes
  packed per 4096B sector, sector 0 holding a metadata record (npts, ndims, medoid,
  max_node_len, nnodes_per_sector, etc.). This is exactly what `HARNESS_JOBS.md` §3 assumed.
- File-size arithmetic reconciles exactly against this layout for all three corpora (SIFT10M:
  7 nodes/sector, 1,428,573 sectors x 4096 = 5,851,435,008 B; Deep10M: 5 nodes/sector,
  2,000,001 x 4096 = 8,192,004,096 B; MS MARCO: 1 node/sector, 8,831,824 x 4096 =
  36,175,151,104 B — all match the sizes in the Job 4 table above exactly).
- Confirmed by direct parse (`harness/diskann_format.py`, verification script run
  2026-09-16): for each corpus, the vector embedded in `disk.index` at node ids 0, 1, 12345,
  and npts-1 is byte-identical to the corresponding row of the base corpus file used to build
  that index (`base.10M.u8bin`, `base.10M.fbin`, `base.8.83M.fbin`). Degree is stored
  per-node and does vary (e.g. MS MARCO node 0 degree=87, node 12345 degree=90, last node
  degree=31 — not a fixed R=100), confirming adjacency is real per-node data, not padding to
  a constant.
- What actually lacks a separately-stored full-precision copy is the **codes tier**: PQ codes
  live only in `pq_compressed.bin` (8B header + npts x 32B, format also confirmed by direct
  parse), decoded via the 3-section `pq_pivots.bin` (offset table -> pivot tables, per-dim
  centroid, chunk_offsets). Decoding node 0's PQ code and comparing to its true vector gives a
  small nonzero reconstruction error as expected (SIFT10M relative L2 error 13%, Deep10M 21%,
  MS MARCO 62% — consistent with MS MARCO's more aggressive 32B/768-dim compression ratio
  already flagged in Job 4).
- Net effect: all three of DuraVec's simulated tiers (CODES, ADJ, VEC) are already present in
  the existing builds with no rebuild and no `--use_reorder_data` needed. `disk.index` supplies
  VEC + ADJ per node; `pq_compressed.bin`/`pq_pivots.bin` supply CODES independently. The Job 4
  caveat's practical conclusion (tier-separation work needs more than these baseline indexes)
  does not hold for the ablation-harness scope in `HARNESS_JOBS.md`, which only needs to parse
  and mask what's already on disk, not physically re-lay it out.
- MS MARCO's `nnodes_per_sector=1` with `max_node_len=3476` on a 4096B sector wastes 620B/node
  (~15%) to sector padding. That padding is not restorable-tier content, so the ablation
  harness's byte model uses actual stored bytes per tier (vector width, 4 + actual_degree*4,
  32B PQ code), not sector-padded `disk.index` file size, when computing bytes restored.

## Harness measurement 1 — validation against stock (2026-09-16, HARNESS_JOBS.md scope)

Built `harness/` (`diskann_format.py`, `mask.py`, `search.py`, `importance.py`, `absence.py`,
`runner.py`), each with unit tests against a tiny synthetic index built by the real
`build_disk_index` binary (`tests/`, 42/42 passing). Layout facts for the parser and the
multi-medoid entry-point handling were traced directly from `diskann_src` (`disk_utils.cpp`,
`pq.cpp`, `pq_flash_index.cpp`), not inferred or assumed — including that MS MARCO's sharded
build produces 3 candidate medoids (`_medoids.bin`/`_centroids.bin`, ids `[754634, 1715546,
754634]`), selected by exact (not PQ) distance to the query, which stock `search_disk_index`
does and which the harness now reproduces.

**Primary gate (per-query result-set agreement against stock, not just aggregate recall):**
ran the harness's unmasked search over the full 10,000-query set of each corpus at stock
params (`L=50, W=4, k=10`) and diffed each query's returned id set against
`logs/job4_results/*_search_50_idx_uint32.bin` (Job 4's own saved stock output).

| Corpus | Exact id-set match | Mean id-set overlap | Harness recall@10 | Stock recall@10 | Gap | Wall-clock (10K queries) |
|---|---|---|---|---|---|---|
| SIFT10M | 99.94% | 0.9999 | 98.009% | 98.010% | 0.001pt | 74.7s |
| Deep10M | 99.90% | 0.9996 | 95.506% | 95.542% | 0.036pt | 88.3s |
| MS MARCO | 99.70% | 0.9995 | 83.521% | 83.534% | 0.013pt | 65.8s |

All three pass both the primary (id-set) and backstop (0.5-point aggregate recall tolerance)
gates by a wide margin. The small residual (<0.3% of queries with a non-exact id set) is
consistent with floating-point tie-breaking on near-equal distances — the same effect already
noted for the SIFT10M/Deep10M ground-truth cross-check against the published reference GT — not
a harness defect. Command and full JSON reports: `scripts/validate_harness.py`, one run per
corpus, output at `pilot/results/measurement1_{sift10m,deep10m,msmarco}.json`.

Note on `disk.index` cold-mmap cost: a first, cold-page-cache pass over a query set is roughly
45x slower (370ms/query observed on SIFT10M) than a warm-cache pass (8ms/query) purely from
first-touch page faults on the multi-GB memory-mapped file, not from the search algorithm.
Relevant for scheduling later sweeps: touching a corpus's files once warms the whole run.

## Harness measurement 2 — the H1 read, SIFT10M (2026-09-16, HARNESS_JOBS.md scope)

SIFT10M, all three tiers restored together, group size 1,000 nodes (10,000 groups), evaluated
on the 5,000-query eval half of the workload/eval split (seed 1234; the other 5,000 are
reserved for the traversal-frequency proxy, not used here). Sweep: coarse pass at 11 fractions
(0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0), then a second pass refined in [0.75,
1.0] once the coarse pass showed the H1-threshold crossing sits there, not near the low-fraction
knee originally expected — see finding below. Sequential and random (5 seeds: 101–105, per
PREREGISTRATION.md's ≥5-seed commitment) run alongside in-degree as the two required baseline
orders. 112 sweep points total (77 coarse + 35 refined), each a JSON provenance record under
`pilot/results/measurement2/sift10m/`, resumable (re-running the same fraction/order/seed loads
the cached record instead of recomputing). Intact recall (reference, all orders agree at
fraction=1.0): 0.9799, matching Job 4's stock full-query-set number closely despite using a
different, smaller 5K eval subset — consistent, not a coincidence to worry about.

**Finding: the low-fraction knee is real and large, but it is not where the H1 threshold sits.**
At low restored fractions, in-degree order is dramatically more byte-efficient than random or
sequential — e.g. at 1% of bytes restored, in-degree reaches 4.3x random's raw recall (0.0208 vs
~0.001), and at 2% roughly 15-20x. Its absence ceiling is more than double random's at the same
fraction (0.1244 vs 0.1014 at 5%), confirming in-degree genuinely front-loads query-relevant
(not just well-connected-in-general) nodes. This is a clean, unambiguous concentration effect —
the mechanism H1 posits is real.

But PREREGISTRATION.md's H1 threshold is 0.9 *normalized* recall (= 0.8819 raw recall@10 here,
since intact = 0.9799), and reaching that high a fraction of intact recall requires restoring
nearly the whole graph under every order tested, which is exactly where the ordering advantage
washes out: by 90% of bytes restored, in-degree's raw recall (0.9386) and random's (0.8809 avg
of 5 seeds) have converged far more than at low fractions. Interpolated crossing points
(`scripts/analyze_measurement2.py`, linear interpolation between the two bracketing measured
points, refined range spaced at 0.05-fraction / ~0.28GB steps):

| Order | Bytes to reach 0.9 normalized recall |
|---|---|
| sequential | 5.067 GB |
| random (avg of 5 seeds) | 5.082 GB |
| in-degree | 4.485 GB |

Better of sequential/random: sequential, 5.067 GB. **H1 ratio (in-degree / better-of-seq-random)
= 4.485 / 5.067 = 0.885 — fails the ≤ 1/3 (0.333) threshold by a wide margin, not narrowly.**
Curve data: `pilot/results/measurement2_sift10m_curve.csv`; plot:
`pilot/results/measurement2_sift10m_curve.png`.

Per HARNESS_JOBS.md 8's instruction not to tune toward the threshold: no parameter was adjusted
after seeing this. Per PREREGISTRATION.md, H1 needs to hold on at least 2 of 3 corpora, and this
measurement covers only SIFT10M with the in-degree order — hop-distance and traversal-frequency
orders (measurement 3, other corpora) haven't been tried yet and PREREGISTRATION.md explicitly
allows the best-performing importance order to be chosen post hoc from among those evaluated. So
this is a real miss on SIFT10M/in-degree specifically, not yet a verdict on H1 overall.

## Harness measurement 3 — Deep10M, all 5 orders (2026-09-16, HARNESS_JOBS.md scope)

Extended the harness to hop-distance (multi-source BFS, generalizes cleanly to MS MARCO's
multiple candidate medoids — see `harness/importance.py: compute_hop_distance`) and
traversal-frequency (5,000-query workload replay, disjoint from the 5,000-query eval set;
97.3% of nodes never visited by the workload, confirming the earlier concern about an enormous
zero-visit tie class was correct — resolved by the recorded random tie-break seed and
`order_by_score`'s explicit tie-break parameter). Added both to SIFT10M (`pilot/results/
measurement2/sift10m/`, 32 new points; the existing sequential/in-degree/random points from
measurement 2 were correctly reused, not recomputed — this also surfaced and fixed a bug where
`RunSpec`'s content hash included the free-text `label` field, which would have silently broken
resumability across scripts with different wording for identical runs; fixed in `harness/
runner.py`, covered by a new test). Then ran the full 5-order sweep (sequential, in-degree,
hop-distance, traversal-frequency, random x5 seeds) on Deep10M: coarse pass (11 fractions, 99
points, 76.6 min) followed by a refine pass once the coarse pass located the crossing region
(9 fractions in [0.72, 0.95], 81 points, 63.7 min, chosen the same way as SIFT10M's refine —
locate the region from coarse data first, don't assume it transfers from the other corpus).

**In-degree remains the strongest order, by a wide margin over the newly-added hop-distance and
traversal-frequency — on this corpus, both underperform in-degree and are barely distinguishable
from random near the threshold crossing.** Intact recall 0.9562 (reference, 9 runs agree
exactly), H1 target 0.8605 raw recall@10.

| Order | Bytes to reach 0.9 normalized recall |
|---|---|
| sequential | 7.347 GB |
| hop-distance | 7.326 GB |
| traversal-frequency | 7.349 GB |
| random (5-seed avg) | 7.364 GB |
| in-degree | 6.270 GB |

Better of sequential/random: sequential, 7.347 GB. **H1 ratio (best importance order = in-degree
/ better-of-baseline) = 6.270 / 7.347 = 0.853 — fails the ≤ 1/3 threshold**, similar margin to
SIFT10M's 0.885. Hop-distance and traversal-frequency's crossings (7.326 GB, 7.349 GB) are
statistically indistinguishable from sequential's (7.347 GB) — at the fractions relevant to this
threshold, neither proxy beats doing nothing clever at all, on this corpus. No parameter was
adjusted after seeing this. Curve data: `pilot/results/measurement3_deep10m_curve.csv`; plot:
`pilot/results/measurement3_deep10m_curve.png`; analysis: `scripts/analyze_measurement3.py`.

**Running tally: 2 of 2 corpora measured so far fail H1 at the ≤1/3 threshold, both by a wide
margin (~0.85–0.89), both with in-degree as the best available order.** PREREGISTRATION.md
requires passing on at least 2 of 3 corpora, so MS MARCO's result is not yet decisive on its own
either way — if it also fails, H1 fails outright and, per §2, the project stops as specified.
MS MARCO is next.

## Harness measurement 3 — MS MARCO, and the H1 verdict (2026-09-16, HARNESS_JOBS.md scope)

Same 5-order sweep as Deep10M (coarse 11 fractions/99 points/57.5 min, refine 9 fractions in
[0.72, 0.95]/81 points/44.4 min). Multi-medoid entry-point handling (3 candidates,
`compute_hop_distance`'s multi-source generalization) exercised for real here for the first
time — no issues.

**None of the importance orders show a meaningful advantage over sequential or random on this
corpus.** Intact recall 0.8359 (reference, 9 runs agree exactly, matches Job 4's flagged 83.5%
figure — the low recall from the aggressive 32B/768-dim PQ compression, not a new issue). H1
target: 0.7523 raw recall@10.

| Order | Bytes to reach 0.9 normalized recall |
|---|---|
| in-degree | 26.084 GB |
| sequential | 26.102 GB |
| traversal-frequency | 26.185 GB |
| random (5-seed avg) | 26.247 GB |
| hop-distance | 26.804 GB (worse than doing nothing clever) |

**H1 ratio (best importance order = in-degree / better-of-baseline = sequential) = 26.084 /
26.102 = 0.9993 — in-degree needs 99.93% of the bytes sequential needs. This is not a narrow
miss, it is essentially zero measured effect.** The restoration curve (`pilot/results/
measurement3_msmarco_curve.png`) makes this visually unambiguous: all five orders collapse onto
one line — recall is proportional to bytes restored regardless of which bytes, on this corpus.
No parameter was adjusted after seeing this.

**Plausible mechanism, not yet verified by a dedicated measurement:** MS MARCO's bytes are ~88%
vectors (§4.1 byte-composition table cited in HARNESS_JOBS.md), versus SIFT10M's ~23%/Deep10M's
intermediate split, where adjacency dominates. Node-importance orders (in-degree, hop-distance,
traversal-frequency) are fundamentally about which nodes are structurally/navigationally
important — that lever mainly pays off through adjacency (better navigation to more of the true
neighbor set). If recall here is instead bottlenecked by simply how many nodes have their
full-precision vector available at all for exact re-ranking, a per-node importance order doesn't
target the actual constraint, because it restores a node's vector exactly when it restores that
node's adjacency, and the two tiers' byte costs aren't targeted independently. This is exactly
what measurement 4 (tier-separated curves, not yet run) is designed to distinguish — and
HARNESS_JOBS.md §7.4 already flagged the expectation that MS MARCO would show "a large effect"
from tier-first restoration specifically, which is a different, not-yet-tested question from
whether *node* importance order helps. Recorded here as a hypothesis for measurement 4, not a
finding.

### H1 verdict across all three corpora

| Corpus | H1 ratio (best importance order / better-of-baseline) | Result |
|---|---|---|
| SIFT10M | 0.885 | fails |
| Deep10M | 0.853 | fails |
| MS MARCO | 0.9993 | fails (no measurable effect) |

**H1 fails on all three corpora against the ≤1/3-of-bytes threshold fixed in
PREREGISTRATION.md §2 before any measurement was run.** PREREGISTRATION.md requires passing on
at least 2 of 3 corpora for H1 to hold; it holds on zero. Per §2's own pre-committed decision
rule: *"If H1 fails: the project stops. There is no ordering to exploit, and the second and
third contributions have no basis."* Per §5.4, a threshold missed narrowly is reported as missed
— these misses (0.85–0.89 on two corpora, ~1.0 on the third) are not narrow. No entries were
made to §6 (post-hoc clarifications) because no ambiguity requiring one was encountered; the
threshold as written was directly measurable as specified.

What *is* real and consistently measured across SIFT10M and Deep10M (not MS MARCO): in-degree
order gets substantially more recall per byte than random or sequential at low-to-moderate
restored fractions (up to ~20x at 1-2% of bytes on SIFT10M) — the concentration mechanism H1
posits genuinely exists structurally. It just doesn't clear the specific 90%-of-intact-recall /
33%-of-bytes bar this pre-registration set, because that bar sits in the saturation region where
the ordering advantage has mostly washed out, on every corpus tested. Measurement 4 (tier-first
curves, run below) and measurement 5 (B5 beam-width/absence-navigation decomposition, not yet
run) don't change this go/no-go outcome per PREREGISTRATION.md §2 once H1 itself has failed, but
per HARNESS_JOBS.md §4 they keep diagnostic value for the write-up, which is why measurement 4
was run anyway at the PI's request.

This is a measured result reported as specified, not a recommendation on what to do next — that
call belongs to the PI per PREREGISTRATION.md's front matter.

## Harness measurement 4 — tier curves, run for the write-up (2026-09-17)

Run after H1 failed on all three corpora (measurement 3); per the PI's request, for the write-up's
diagnostic value, not as a go/no-go gate. Three cumulative stages restoring each tier at 100%
across the whole graph (a single-group mask, not a per-node fraction sweep like measurements 2/3):
codes-only, codes+adjacency, then everything (intact). Same 5,000-query eval subset as
measurements 2/3. `scripts/run_measurement4.py`; plot at
`pilot/results/measurement4_tier_contributions.png`.

**Codes-only recall is exactly 0.0000 on all three corpora, and this is expected, not a bug.**
With no `ADJ` anywhere, expanding the entry point yields zero neighbors (`miss_count` = 1 per
query, `num_expanded` = 1) — there is nothing to navigate with, only something to score. Absence
ceiling is 1.0000 at this stage on all three corpora (every true neighbor technically "has
CODES" everywhere), so the entire recall deficit here is broken navigation, not absence — the
cleanest possible confirmation that §5.3's absence/navigation decomposition and §4's ADJ-miss
semantics behave exactly as specified.

| Corpus | CODES bytes | ADJ bytes | VEC bytes | codes-only | +adjacency | +vectors (intact) | Δ_ADJ | Δ_VEC |
|---|---|---|---|---|---|---|---|---|
| SIFT10M | 0.32 GB (5.7%) | 4.04 GB (71.6%) | 1.28 GB (22.7%) | 0.0000 | 0.7190 | 0.9799 | +0.7190 | +0.2609 |
| Deep10M | 0.32 GB (3.9%) | 4.04 GB (49.3%) | 3.84 GB (46.8%) | 0.0000 | 0.6517 | 0.9562 | +0.6517 | +0.3045 |
| MS MARCO | 0.28 GB (1.0%) | 2.11 GB (7.2%) | 27.13 GB (91.8%) | 0.0000 | 0.5399 | 0.8359 | +0.5399 | +0.2960 |

**This contradicts the literal expectation in HARNESS_JOBS.md §7.4 as stated, and that's worth
saying plainly rather than smoothing over.** The prediction was "a large effect on MS MARCO...
and very little on SIFT" from unlocking vectors, reasoning from byte share (92% vs 23%). In
absolute recall points, Δ_VEC is nearly identical across all three corpora (0.261, 0.305, 0.296)
despite a 4x range in the vector tier's byte share — not "large vs very little," essentially
flat. Two ways this can be sliced that come closer to the stated intuition, reported for
completeness rather than to rescue the prediction:
- *Relative to intact recall*, MS MARCO's deficit without vectors is the largest (35.4% of
  intact) versus Deep10M (31.9%) and SIFT10M (26.6%) — the ordering the prediction implied, just
  a modest gradient rather than a stark one.
- *Bytes per recall point are wildly different in the other direction*: SIFT10M's vector tier
  buys its +0.261 recall for 1.28 GB (≈4.9 GB/recall-point); MS MARCO's buys +0.296 for 27.13 GB
  (≈91.7 GB/recall-point) — SIFT's vectors are roughly 19x more byte-efficient, the opposite of
  "large effect."

HARNESS_JOBS.md §7.4's explicit sanity check — "if SIFT shows a large tier-separation gain,
suspect the harness, not the data" — is **not** triggered: SIFT10M shows the *smallest* absolute
Δ_VEC of the three (0.261, versus 0.305 and 0.296), which is consistent with the harness being
correct. The finding is that the predicted effect size didn't materialize as strongly as
expected in the metric (absolute recall) the prediction was probably imagining, not that the
harness got something backwards.

## Status as of this entry (2026-09-16)

All four PILOT_JOBS.md jobs complete: persistent storage, DiskANN pin + build (with one small,
documented patch), SIFT10M/Deep10M/MS MARCO corpora and queries prepared, all three disk indexes
built, ground truth computed and cross-validated, baseline recall/latency/size characterized and
flagged where relevant. Data-prep scope for this spec is done; the ablation harness, masking
logic, restoration curves, and Ceph-touching work are explicitly out of scope here per the job
file and belong to a separate spec.

## Harness measurement 5 — B5 beam-width sweep, run for the write-up (2026-09-17)

Run after H1 failed on all three corpora (measurements 2–3); per the PI's request, for the
write-up's diagnostic value, alongside measurement 4. Random order (seed 101, group size 1,000),
restored fractions 0.02/0.05/0.10 chosen from measurement 2/3's data as the range where measured
recall sits well below the absence ceiling (i.e. where "broken navigation," not absence, is the
dominant deficit and search effort has something to recover). Deliberate 1,000-query subsample of
the 5,000-query eval set (documented, not silent) — this is a characterization sweep, not a
go/no-go gate, so the full query-set cost isn't warranted. `scripts/run_measurement5.py`; plot at
`pilot/results/measurement5_b5_sweep.png`.

**Correction to an assumption made while building `harness/search.py`, caught by this
measurement's own first run.** The plan (and the module's own docstring at the time) held that
DiskANN's beam width (`W`, candidates expanded per hop) only batches I/O and can't change which
nodes end up expanded, since the search runs to exhaustion of the `L`-bounded candidate list
regardless of batch size — so a "W shouldn't matter" sanity check was built into this script.
**It wasn't flat**: at SIFT10M frac=0.05, recall rose from 0.0315 (W=2) to 0.0361 (W=16), a small
but real and monotonic effect, confirmed at every fraction on all three corpora. Mechanism (traced
after the surprise, not before): the candidate list is capacity-bounded, and an already-expanded
entry still occupies a slot until evicted by a strictly better later insertion. A larger `W`
expands several close candidates together off one shared snapshot of the list before any of their
discoveries feed back in — the same "explore more broadly before committing" mechanism that makes
wider beams help in beam search generally. So HARNESS_JOBS.md's literal "beam width" (`W`) is a
real, if secondary, search-effort knob after all; the harness now sweeps both `L` and `W` at each
fraction, not just `L`.

**Both knobs recover real broken-navigation, `L`'s effect is 2–5x larger than `W`'s, and the
recoverable fraction differs sharply by corpus:**

| Corpus | frac | ceiling | broken-nav at L=50 | broken-nav at L=400 | % recovered | broken-nav at W=2 | broken-nav at W=16 | % recovered |
|---|---|---|---|---|---|---|---|---|
| SIFT10M | 0.10 | 0.1005 | 0.0144 | 0.0015 | 90% | 0.0156 | 0.0111 | 29% |
| Deep10M | 0.10 | 0.0971 | 0.0157 | 0.0012 | 92% | 0.0159 | 0.0119 | 25% |
| MS MARCO | 0.10 | 0.0988 | 0.0382 | 0.0157 | 59% | 0.0399 | 0.0308 | 23% |

**MS MARCO shows a genuinely different, harder-to-recover regime, not just a scaled-down version
of SIFT10M/Deep10M's pattern.** At every fraction tested, its broken-navigation gap is 2–4x wider
than SIFT10M/Deep10M's at the same L, and less of it closes with more search effort (59% recovered
by L=400 at frac=0.10, vs 90%+ for the other two). **At frac=0.02, MS MARCO's recall is exactly
0.0000 at every single L and W value tested (mean_hops pinned at exactly 2.0 throughout)** —
search-effort recovery has a floor: if the restored subgraph is too sparse/disconnected near the
entry point, no amount of extra beam width or search-list size helps, because there is nothing
reachable to find regardless of how hard you look. SIFT10M/Deep10M don't hit this floor at 2% in
this sweep (both still show partial recovery). Plausible cause, not yet directly measured: MS
MARCO's per-node degree is far less uniform than SIFT10M/Deep10M's (min degree 2 observed in the
harness-validation spot check, vs both other corpora sitting close to the R=100 cap almost
everywhere), so a random 2% node subset is more likely to strand the entry point's neighborhood
on MS MARCO than on a corpus whose Vamana degree distribution is more uniform.

Per PREREGISTRATION.md §4, B5 "does not by itself fail H2" and doesn't change H1's already-decided
outcome; it's characterization, consistent with why it was run only after the PI asked, for the
write-up. A strong B5 (SIFT10M/Deep10M's 90%+ recovery) narrows DuraVec's potential margin over a
recall-first-with-wider-search baseline, which is exactly the kind of result PREREGISTRATION.md §4
says must be reported regardless of whether it's flattering.

## Status as of this entry (2026-09-17)

HARNESS_JOBS.md measurements 1–5 complete: harness built and validated against stock (measurement
1, all three corpora pass); H1 measured on all three corpora (measurements 2–3) and fails on all
three against the pre-registered ≤1/3-of-bytes threshold — per PREREGISTRATION.md §2, H1 does not
hold and the project's own pre-committed rule says to stop; tier curves (measurement 4) and the B5
beam-width sweep (measurement 5) run afterward at the PI's request for the write-up's diagnostic
value, not as further gates — notably, measurement 5's own first run caught and corrected a wrong
assumption made while building the search harness (beam width W does affect recall, not just
search list size L). This is a measured outcome, not a recommendation — the decision on how to
proceed belongs to the PI.
