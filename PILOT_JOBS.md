# Task: DuraVec pilot — data preparation jobs

You are setting up and running the data-preparation jobs for the DuraVec pilot. These
jobs produce the corpora, indexes, and ground truth that every later measurement depends on.
They are long-running and mostly unattended.

**Read this whole file before running anything.** Then propose a plan and wait for my approval
before starting any job that takes more than a few minutes.

---

## Environment

- Single node, one NVIDIA A100, Linux. Everything runs here — no cluster, no distributed setup.
- Check available RAM, disk, and CUDA version before planning anything, and tell me what you find.
- Persistent storage: confirm with me which path survives instance teardown. **All outputs go
  there**, never to ephemeral instance disk.
- Long jobs run detached (tmux or nohup) with output teed to a log file, so a dropped SSH
  session never kills a build.

## Scope — what you are and are not doing

**In scope:** obtain corpora, build indexes, verify them, compute ground truth, generate
embeddings, record provenance.

**Out of scope, do not start:** the ablation harness, masking logic, restoration curves, any
recall-vs-restored-fraction measurement, anything touching Ceph. Those are separate work with a
separate spec. If you finish early, stop and tell me.

---

## Jobs

Priority order. Job 1 is the long pole and should be started first even though it finishes last.

### Job 1 — MS MARCO passage embeddings (GPU)

~8.8M passages through a sentence embedding model. This is the only job that genuinely needs
the A100.

**Before generating anything, search for a precomputed embedded version of MS MARCO passages
for a standard model and tell me what you find.** Downloading beats regenerating. Only generate
if nothing suitable exists, and confirm the model choice with me first — the embedding model
determines the dimensionality of the corpus and therefore every downstream byte-budget number.

If generating: batch on GPU, checkpoint output shards as you go so an interruption costs one
shard rather than the run, and report throughput after the first few shards so we can estimate
total time.

Output: float32 vectors in a format the index builder reads directly, plus a held-out query set.

### Job 2 — Index builds: SIFT 10M and Deep 10M

Build disk-resident Vamana/DiskANN indexes over the first 10M points of SIFT1B and Deep1B.

- Get the DiskANN source, pin one upstream commit, and record the commit hash. We will fork
  this later, so the starting point must be reproducible.
- Build parameters: use values consistent with the published DiskANN configurations for these
  corpora. Propose R, L, and PQ byte budget and justify them before building; do not silently
  pick defaults.
- CPU build is expected and fine. Report wall-clock build time.

Build MS MARCO's index too, once Job 1 produces vectors.

### Job 3 — Ground truth

Exact k-NN at **k=10** for every query set against the full corresponding corpus.

- Use a batched/blocked GPU implementation. Do not write a naive loop.
- Verify correctness on a small subset against a brute-force CPU computation before running the
  full job.
- This is the job most likely to be needed unexpectedly later, so it must be complete and
  correct, not approximate.

### Job 4 — Baseline characterization (short, but do not skip)

For each built index, with all data present and stock search parameters, measure and record:

- **recall@10 against the ground truth** — compare to published numbers for these corpora and
  flag any discrepancy loudly. A mis-parameterized build silently poisons every later figure.
- **P99 query latency** on an otherwise idle machine. This is the denominator for the TTRS
  latency bound and the query timeout, so it needs to be a clean measurement.
- Index size on disk, broken down by tier if the format exposes it.

---

## Provenance log — the real deliverable

Maintain a single file, `pilot/provenance.md`, committed to the repo, recording for every
corpus:

- source and exact version/subset of the data, with how it was obtained
- embedding model, if applicable
- DiskANN commit hash and full build parameters
- build wall-clock time and hardware
- intact-index recall@10 and P99 latency, with the search parameters they were measured at
- ground-truth method and verification result
- absolute paths to every artifact

Every later figure normalizes against the intact-index recall recorded here, so this file is
cited, not just archived. Append as each job completes — do not reconstruct it at the end from
shell history.

---

## Working style

- Propose the plan first. Wait for approval before long jobs.
- One job at a time to completion where they contend for the GPU; Job 2's CPU builds may overlap
  with GPU work.
- Everything scripted and re-runnable. If I have to re-run a corpus in three weeks, it should be
  one command, not a reconstruction.
- Scripts take paths and parameters as arguments. No hard-coded absolute paths.
- Report failures immediately with the actual error. Do not silently fall back to a smaller
  dataset, fewer points, or different parameters to make something succeed.
- If a published recall number and our measured number disagree, stop and tell me. Do not tune
  parameters to close the gap without discussing it.
