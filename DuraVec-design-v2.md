# DuraVec — Design Document and Implementation Plan

**Companion to the research proposal (rev. 3), September 2026**
Vladislav Esaulov · advised by Lipeng Wan · Georgia State University

This document is the engineering counterpart to the proposal. The proposal argues *why*
graded recovery is worth a paper; this one specifies *what gets built*, in what order, and
which decisions are still open. Where the two disagree, the proposal is the claim of record
and this document is the implementation of it.

---

## 1. Scope

### 1.1 Goals

DuraVec is a layer over a disk-resident Vamana/DiskANN index that turns the interval between
a serving failure and full restoration into a design target. It must:

- **G1.** Restore index state in an order chosen to maximize recall gained per byte moved.
- **G2.** Answer queries over a partially restored index without stalling on missing data.
- **G3.** Report per-query recall estimates and provenance so degraded results are usable.
- **G4.** Reconstruct the pre-crash graph exactly after a process crash.
- **G5.** Measure all of the above as time-to-recall-SLO under a latency bound.

### 1.2 Non-goals

- Not a new ANN algorithm. Vamana search and build are used as-is.
- Not a distributed query engine. One serving node holds one index shard; multi-shard routing
  is out of scope, and a replicated serving tier is a baseline to argue against, not to build.
- Not a new logging discipline. Graph-mutation logging exists (Weaviate, P-HNSW); we implement
  it for Vamana because we need it, and claim nothing for it.
- Not a production system. No auth, no multi-tenancy, no online reconfiguration.
- **Not an HNSW implementation.** HNSW appears only in the offline pilot measurements.

### 1.3 Success criteria

The build is successful if it can produce Figures A–D of the proposal on real hardware, with
baselines B0–B4 and ablations A1–A6 running as switches in the same binary rather than as
separate systems. Anything not needed for those figures is out of scope by definition.

---

## 2. Background: what DiskANN gives us

The design leans on specifics of the DiskANN on-disk format, so they are worth stating.

A built SSD index is a set of files under one prefix: `_pq_pivots.bin` (codebook),
`_pq_compressed.bin` (one compressed code per vector, held in memory), and `_disk.index`
(the graph). In `_disk.index`, each node occupies a slot inside a 4 KiB sector holding its
full-precision vector followed by its adjacency list — vector and neighbors are **co-located
in one block**, which is what makes a single random read serve both distance computation and
expansion.

Search (`PQFlashIndex::cached_beam_search`) is beam search: candidates are scored cheaply
using in-memory PQ codes, the best `W` unvisited candidates are read from SSD per iteration,
and full-precision vectors from those reads re-rank the final list. There is already a node
cache (`num_nodes_to_cache`, typically seeded near the medoid) — the closest thing in the
codebase to what we are generalizing.

The in-memory dynamic index (`Index<T>`) supports concurrent `insert_point`, `lazy_delete`,
and `consolidate_deletes`; `inter_insert` is where back-edges are written into existing
nodes' adjacency lists during robust pruning. That function is our logging hook.

**The co-location problem.** DiskANN's block layout is exactly wrong for us: it binds the
adjacency tier to the vector tier, so you cannot restore one without the other. Tier
separation (§4.1) is therefore not a convenience — it is the enabling change, and it means
DuraVec's on-disk format diverges from stock DiskANN. This is the single largest piece of
engineering in the project and the first thing to build.

---

## 3. Architecture

```
                        ┌──────────────────────── serving node ──────────────────────┐
   queries ──────────►  │  Query driver  ─►  Serving shim                            │
                        │                     ├─ tier state table (§4.3)             │
                        │                     ├─ modified beam search (§4.4)         │
                        │                     ├─ miss policy: skip | fetch (§4.5)    │
                        │                     └─ recall estimator (§4.6)             │
                        │                            │ promote(unit)                 │
                        │                            ▼                               │
                        │   Recovery planner (§4.2) ─────► I/O budget (token bucket)  │
                        │        │ ordered restore units        │ shared              │
                        │        ▼                              ▼                     │
                        │   Tier store: local NVMe blocks  ◄── RADOS fetcher (§4.7)   │
                        │        ▲                                                    │
                        │   Mutation logger (§4.8) ──► log objects                    │
                        └────────────────────────────┬───────────────────────────────┘
                                                     ▼
                                      ┌──── Ceph pool (existing cluster) ────┐
                                      │  manifest · codes/ · adj/ · vec/     │
                                      │  logs/                               │
                                      └──────────────────────────────────────┘
```

Six components. The checkpointer (offline, §4.1) produces what the pool holds; the planner
(§4.2) decides what comes back and when; the shim (§4.3–4.6) serves whatever has arrived;
the fetcher (§4.7) moves bytes; the logger (§4.8) handles process crashes; the harness (§5)
measures all of it.

---

## 4. Component design

### 4.1 Checkpointer and tiered, hub-aware layout

**Input:** a stock DiskANN index. **Output:** a checkpoint in the Ceph pool.

Three steps.

**Step 1 — rank nodes by importance.** Compute a proxy score per node and sort descending to
get a `rank_id` for every `orig_id`. Proxies (all cheap, all computed once at checkpoint time):

| Proxy | Cost | Notes |
|---|---|---|
| In-degree | One pass over adjacency | Cheapest; strong hub correlate |
| Hop distance from medoid | One BFS | Captures entry region |
| Traversal frequency | Replay a sample query workload, count visits | What a warm-up policy would use — the H2 baseline, so it must be implemented |
| Approximate betweenness | Sampled shortest paths | Expensive; include only if the pilot shows it is worth it |

The permutation is stored in the manifest. All internal identifiers are **rank-space**; the
shim translates at the API boundary.

**Step 2 — group nodes.** Group `g = rank_id / G`. Two group sizes: small priority groups
(target ~1 MiB objects) for the top-ranked prefix, larger ones (target ~8–16 MiB) for the
tail, to keep object count manageable. The exact split is a tuning parameter fixed after the
pilot. Because grouping is arithmetic in rank space, node→group is a shift, not a lookup —
this is what keeps the hot path free of per-node metadata (§4.3).

**Step 3 — write tier-separated objects.**

```
<idx>/<epoch>/manifest
<idx>/<epoch>/codes/<group>     PQ codes for the group's nodes, in rank order
<idx>/<epoch>/adj/<group>       adjacency lists, neighbor IDs in rank space
<idx>/<epoch>/vec/<group>       full-precision vectors
```

The manifest holds: index parameters (R, L, PQ bytes, dimension, metric), PQ pivots, medoid,
the permutation map, the group table (group id, tier, byte size, node count, estimated recall
gain), the epoch, and the log LSN watermark the checkpoint is consistent to.

**Byte budget** — why tier separation pays, and where it does not:

| Corpus | d | Codes (32 B) | Adjacency (R=64, 4 B) | Vectors (fp32) | Vector share |
|---|---|---|---|---|---|
| SIFT 100M | 128 | 3.2 GB | 25.6 GB | 51.2 GB | 64% |
| Deep 100M | 96 | 3.2 GB | 25.6 GB | 38.4 GB | 57% |
| MS MARCO 8.8M | 768 | 0.3 GB | 2.3 GB | 27.0 GB | 91% |
| Text-to-Image 100M | 200 | 3.2 GB | 25.6 GB | 80.0 GB | 73% |

Codes are 3–10% of the footprint on every corpus, so restoring all codes first is nearly free
and gives a globally scorable (if un-expandable, un-re-rankable) index almost immediately.
The vector tier dominates and dominates hardest at high dimension — which is why MS MARCO is
the corpus where tier-first restoration should look best, and SIFT the one where it looks
weakest. Both belong in the evaluation; reporting only the flattering one would be dishonest.

**Incrementality.** Only groups whose nodes changed since the last epoch are rewritten;
unchanged groups are referenced by the new manifest at their old epoch. Recovery therefore
reads a mixed-age set of objects, and checkpoint cost scales with churn rather than index size.

### 4.2 Recovery planner

State: a priority queue of restore units `u = (group, tier)` keyed by `ĝ_u / b_u`, where `b_u`
is the byte size from the manifest and `ĝ_u` is the estimated recall gain.

**Gain estimates** come from the offline ablation curves (the pilot's output): for each tier
and each importance decile, a marginal-gain-per-byte value, interpolated by restored fraction.
These are static per (corpus, index config), loaded from the manifest. No online learning.

**Dependencies.** A node's adjacency is only useful once its own code is present (you cannot
reach a node you cannot score); a vector is only useful once the node is reachable. The queue
enforces `codes(g) → adj(g) → vec(g)` per group, but not across groups.

**The local-model problem.** The gain table is local: it scores a unit by its own tier and
importance decile. But a hub group's adjacency is worth nothing until enough of its neighbors'
codes are present, so a purely local score will systematically *overvalue isolated hub and
bridge groups* — precisely the units it most wants to schedule first. This is the same
non-submodularity noted below, seen from the planner's side, and it is a correctness problem
for the heuristic rather than a theoretical caveat.

Pilot step 2 (proposal §7) measures the size of the effect by restoring high-rank groups in
isolation and again with their neighbor closure. If realized gain falls well short of
predicted gain, the planner needs one of:

- a **closure rule** that co-schedules a group with the neighbor groups supplying most of its
  in-edges, treating the closure as one unit; or
- a **connectivity discount** on predicted gain, scaled by the fraction of a group's neighbors
  whose codes are already restored, recomputed as restoration proceeds.

The discount is cheaper and composes with the existing queue; the closure rule is more faithful
but inflates unit sizes and coarsens ordering. Decide from the measurement, not in advance.

**Why greedy, and what it is measured against.** Recall gain is not submodular in the restored
set — percolation creates complementarities where a bridge group helps only once both sides
are present — so greedy carries no approximation guarantee. We use it anyway, with limited
lookahead over dependent units, and measure it against an evaluation-greedy reference (§5.3)
that picks by *measured* recall gain on a small index. Optimality is neither claimed nor
computed.

**Budget.** A token bucket refilled at the measured rehydration throughput, shared between
planner prefetch and shim-initiated fetches. Query-driven fetches preempt planner work by
holding a reserved share of tokens (initial split 30/70, tunable) so that a burst of misses
cannot starve bulk restoration or vice versa.

**Promotion.** The shim can promote a unit on a miss; promotion moves it to the queue head and
is deduplicated against in-flight fetches.

### 4.3 Tier state and the hot path

The entire restored-state representation is one array:

```cpp
enum Tier : uint8_t { CODES = 1, ADJ = 2, VEC = 4 };

struct GroupState {
    std::atomic<uint8_t> tiers;   // bitmask of Tier
    uint32_t             epoch;
};
std::vector<GroupState> group_state;   // indexed by group id

inline bool has(uint32_t rank_id, Tier t) const {
    return group_state[rank_id >> kGroupShift].tiers.load(
               std::memory_order_relaxed) & t;
}
```

One relaxed atomic load and a shift per check. No per-node bitmap, no allocation, no lock on
the read path; the fetcher publishes with a release store once a group's bytes are durable
locally. At 100M nodes and 64K-node groups this table is ~1,500 entries — it fits in L1.

This is the payoff of arithmetic grouping in rank space. Had we kept contiguous original IDs
and an arbitrary node→group map, every hot-path check would be a cache-missing lookup.

### 4.4 Modified beam search

Changes to `cached_beam_search`, each gated so the stock behavior is recoverable for baselines:

1. **Seeding.** If the medoid's group lacks `CODES`, fall back to the highest-ranked node with
   codes present. Rank order makes this a scan from zero, and in practice rank 0 is restored
   first, so the fallback should never fire outside adversarial ablations.
2. **Scoring.** A candidate without `CODES` is not scorable and is dropped.
3. **Expansion.** A candidate without `ADJ` is a *miss*: hand it to the miss policy (§4.5).
   If skipped, the node still contributes its own distance to the result list but yields no
   neighbors. Search over a partial index is search over the restored subgraph.
4. **Neighbor filtering.** Neighbor IDs pointing into groups without `CODES` are dropped at
   expansion time rather than enqueued and dropped later.
5. **Re-ranking.** Results whose `VEC` is present are re-ranked exactly; the rest keep their
   PQ distance. The count of code-only results is recorded per query — it is both an estimator
   feature and a headline diagnostic.
6. **Termination.** Unchanged (list exhausted or `L` reached), plus a wall-clock deadline that
   forces return with whatever is in hand.

Two invariants the correctness harness must check: **search never blocks on a missing unit
unless the miss policy explicitly chose to fetch**, and **with all tiers present, results are
bit-identical to stock DiskANN**.

### 4.5 Miss policy

Per miss, with a per-query deadline:

```
on_miss(unit u, query q):
    if now + eta(u) > q.deadline:          return SKIP
    if predicted_gain(u, q) < threshold:   return SKIP      # promote anyway
    return FETCH                                            # and promote
```

`eta(u)` is an EWMA of recent fetch latency for that tier, separated by source (local NVMe
vs. pool) since they differ by orders of magnitude. `predicted_gain` is a cheap proxy: the
rank of the missing candidate in the current result list, and its distance margin below the
current k-th result. A miss on something that would not have made the top-k is not worth a
round trip.

Ablation A4 pins this to always-skip or always-fetch, which is precisely how B3 and B4 (the
blocking incumbents) are realized in the same binary.

### 4.6 Recall estimator

**Output** per query: estimated recall@10 ∈ [0,1], a degraded flag, and the restored-state
epoch and tier-fraction vector as provenance.

**Features.** Start from DARTH's published feature set for mid-search recall prediction, which
is the closest validated design, and add the features that describe partial restoration. DARTH
uses eleven: search step, distance calculations, and result-set updates (search progress);
distances to the first, current-closest, and current-furthest neighbor; and the mean, variance,
median, 25th and 75th percentiles of result-set distances. Their ablation finds the
search-progress features load-bearing — neighbor-distance features alone perform far worse —
and reports search step, closest neighbor, first neighbor, result-set updates, and distance
variance as the highest-importance individual features.

DuraVec adds **loss features**, which have no analogue in DARTH because a complete index loses
nothing: number of misses skipped and their best rank in the current result list; fraction of
results ranked by code only; distance margin between the k-th result and the best skipped
candidate; and restored fraction per tier. Framing the set as "DARTH's features plus loss
features" is both more defensible and easier to review than presenting a parallel invention.

**Model.** A small gradient-boosted tree (~100 trees, depth ≤4), trained offline on ablated
indexes where true recall is known, calibrated with isotonic regression, exported as a flat
array and evaluated inline.

**Inference budget.** DARTH measures ~0.03 ms per single-input prediction for eleven features
with LightGBM on one core — 30 µs, three times the <10 µs originally assumed here. Two things
work in our favor: we call the estimator **once per query at completion**, where DARTH invokes
it repeatedly mid-search (6 times on average at a 0.80 target, 11 at 0.99), and exporting the
trees to a flat array avoids per-call library overhead. Neither is verified. The budget is
therefore a *measured* target: benchmark the exported model before wiring it in, and if it
lands near 30 µs, decide between shrinking the model and reporting the overhead, rather than
assuming it away. At a `P99` in the low milliseconds even 30 µs is under 3%, so this is a
sizing question, not a feasibility one.

**Why not a global curve.** A single recall(restored-fraction) curve is a population average
and is wrong in the worst way for exactly the queries that matter — the ones whose true
neighbors sit in the unrestored set. Ablation A5 measures how much this matters; if it turns
out not to, the finding is reportable and the estimator simplifies.

**Honesty rule.** Where the model is out of distribution (restored fraction below the lowest
training point), fall back to a conservative lower bound from the global curve rather than
extrapolating. Calibration error is a reported metric, not an internal detail.

### 4.7 RADOS fetcher

Direct `librados` against the existing cluster — no RGW, no S3 gateway, since we want to
inject OSD and host failures underneath and measure what the client sees.

- Async `rados_aio_read` with a fixed in-flight window (start at 32, tune against measured
  throughput).
- Two pools, both exercised: replicated (size 3) and erasure-coded (4+2), because degraded-read
  behavior differs and F2d is about exactly that difference.
- Writes land in a staging file on local NVMe, are `fdatasync`ed, and only then is the tier bit
  published. No torn state is ever visible to the shim.
- Per-fetch telemetry: submit and complete timestamps, bytes, tier, source pool state, and
  in-flight depth at submission. This is the raw material for Figure C, and the depth field is
  what separates a bandwidth-bound regime from a latency-bound one.

**Is recall-per-byte even the right objective?** The planner optimizes recall per byte, which is
correct only if rehydration is bandwidth-bound. If the pool is latency- or IOPS-bound at our
object sizes, then total restore time is set by fetch parallelism rather than by byte volume,
ordering matters much less, and the objective should be recall per *fetch*. Pilot step 3
measures this before P2 commits to the byte-based formulation. The two objectives differ only
when unit sizes vary widely — which they do, by design, since priority groups are small.

### 4.8 Mutation logger (F1 only)

Hook: `Index<T>::inter_insert` and the prune path, i.e. wherever an adjacency list is
finalized.

**Record format — decision: log the whole new adjacency list, not add/remove deltas.**

```
struct MutRecord {
    uint64_t lsn;
    uint32_t node;          // rank space
    uint16_t degree;
    uint32_t neighbors[];   // complete post-mutation list
};
```

Whole-list records are larger (R×4 bytes vs. ~8 per edge change) but idempotent and
order-insensitive on replay: applying the same record twice, or applying records for a node
out of order with respect to *other* nodes, converges to the same graph as long as per-node
LSN order is respected. Delta records require exact replay order globally and make partial
tail loss unrecoverable. For a recovery paper, replay simplicity is worth the bytes — and the
log-volume cost is a reported number (the churn crossover), not a hidden one. Weaviate made
the same call with `ReplaceLinksAtLevel`.

Group commit: records batch into a buffer, flushed on size (1 MiB) or time (10 ms), whichever
first; `fdatasync` before acknowledging the insert. Checkpoint truncates the log at its LSN
watermark.

**Exact-reconstruction harness.** Build with logging on, kill mid-update, replay, and compare
adjacency bit-for-bit against a pre-crash dump. This must pass before any recovery numbers are
reported, because it is the only check that the logger is correct at all.

---

## 5. Measurement harness

The harness is not an afterthought; several of the proposal's claims are only credible if it
is right.

### 5.1 Query driver

**Open-loop, Poisson arrivals at a fixed target rate.** A closed-loop driver would hide exactly
what we are measuring: when queries block, a closed loop simply issues fewer of them and tail
latency looks fine. Open-loop lets the queue build, which is the honest depiction of a blocking
design under recovery.

Per-query log: arrival, start, completion, result IDs, estimated recall, miss counts, restored
epoch. Recall is computed offline against precomputed ground truth. `R(t)` and `P99(t)` are
windowed over `Δ` (start at 1 s) after the failure.

### 5.2 Timed-out queries count as recall 0

A query exceeding 10× the intact index's `P99` is recorded as a miss, not dropped. Without
this, a design that stalls every query would report perfect recall, and the whole comparison
would be meaningless.

### 5.3 Evaluation-greedy reference

On a 1M index with coarse units, restore step-by-step, at each step measuring actual recall for
every candidate next unit and taking the best. Expensive (O(units² × queries) evaluations) and
therefore small-scale only. It bounds how much the cheap proxy leaves on the table.

### 5.4 Failure injection

| Mode | Mechanism |
|---|---|
| F1 process crash | `SIGKILL` the serving process mid-update; NVMe intact |
| F2 node loss | Stop the process, wipe local NVMe state, start a replacement; pool healthy |
| F2d degraded pool | As F2, plus `systemctl stop ceph-osd@N` for the failed host's OSDs |

Each run scripted end to end so a full matrix can be re-run after any change without manual
steps.

---

## 6. Open design questions

Listed because they are genuinely undecided, and each should be closed by measurement rather
than by argument.

1. **Group size and the priority/tail split.** Small objects mean finer ordering and worse
   throughput. Decide from the pilot's throughput-vs-object-size curve.
2. **Which importance proxy.** In-degree is cheapest; traversal frequency is the baseline we
   must beat. If frequency wins outright, H2 is in trouble — better to learn that in the pilot.
3. **Whether adjacency needs its own compressed form.** Truncated adjacency (top-k neighbors by
   some criterion) would let a partial adjacency tier arrive sooner. Attractive, and a scope
   risk; deferred unless the tier curves say it matters.
4. **Budget split between planner and query-driven fetches.** Fixed 30/70 to start; possibly
   adaptive. Not a research question, but it will affect the numbers.
5. **Group-level vs. finer tier state.** Group granularity is what makes the hot path free.
   If ablation A2 shows grouping is losing significant gain, revisit.
6. **Delete and tombstone handling under partial restoration.** A deleted node whose tombstone
   lives in an unrestored group could be served as a live result. Probably: keep the tombstone
   set entirely in the codes tier so it arrives first. Needs to be settled before evaluation.
7. **Whether the gain model needs a connectivity term** (§4.2), and if so whether a discount or
   a closure rule. Decided by pilot step 2.
8. **Do we need a log at all for F2?** Node loss reads from a checkpoint; the log matters only
   for updates after it. If the evaluation ends up F2-dominated, the logger's role shrinks
   further — acceptable, since it is not a claimed contribution.

---

## 7. Language and codebase

DiskANN is C++17. The shim, planner, tier store, and logger are all on the query hot path or
inside DiskANN's own data structures, so they are **C++, in-tree, as a fork**. Introducing an
FFI boundary in the middle of beam search would cost more than it buys and would make the
"bit-identical to stock DiskANN" invariant harder to defend.

Rust is the right choice for the **offline tooling**, where it does not touch the hot path and
the tooling is standalone: the checkpoint writer, the pilot's ablation and curve-fitting
harness, the trace-driven simulator, and the analysis pipeline. That is roughly half the code
by volume and none of it by latency sensitivity.

One caution: two languages means two build systems and a serialization format both sides
agree on. Keep the manifest a plain, versioned, length-prefixed binary format with a written
spec, not whatever either language's ergonomic serializer produces.

---

## 8. Implementation plan

Phases, each with an exit criterion. Nothing starts before P0 passes.

### P0 — Pilot (weeks 0–3, Sep 14 – Oct 4)

*Gate for the entire project. Almost no code that survives into the system.*

| Deliverable | Detail |
|---|---|
| Ablation harness | Load a built index, remove and restore structure by order and tier, measure recall at fixed search params |
| Restoration curves | 10M SIFT and Deep, plus MS MARCO; orders: sequential, random, frequency, in-degree, tier-first; dense sampling near the knee |
| Cross-group dependency test | Restore high-rank groups in isolation vs. with neighbor closure; compare realized against locally predicted gain (§4.2) |
| Search-effort compensation (B5) | Sweep beam width at several restored fractions; decompose recall loss into absence (unrecoverable by search) and broken navigation (recoverable), per proposal §6.2.1 |
| Ceph throughput measurement | 100M-scale volumes of segment-sized objects, healthy and degraded pool; bulk throughput, single-fetch latency, and achieved parallelism recorded separately |
| Trace-driven simulator | Combine curves + byte costs + fetch latencies to simulate `R(t)`, `P99(t)` for DuraVec, B3, B4 |
| Prior-art close-out | Warm-up/lazy-load code paths in Milvus, OpenSearch, Weaviate; adjacency-level logging in PipeANN, OdinANN, DiskANN; repeat literature search |

**Exit:** H1 and H2 as pre-registered in the proposal §7. *Write the thresholds down before
running step 1* — the whole value of a pre-registered gate is lost if the numbers are fixed
after seeing the data.

Two of the P0 deliverables are not gates but design inputs, and both change the system before
it is built: the cross-group test decides whether §4.2 needs a connectivity term, and the
parallelism measurement decides whether recall-per-byte is the right objective at all.

Ordering note: the restoration curves take days and must be in hand for the proposal defense.
Run them first, before the Ceph work; the cross-group test reuses the same harness and follows
immediately.

### P1 — Tiered format and checkpointer (weeks 3–8)

- Node ranking, permutation, grouping.
- Checkpoint writer producing manifest + tier objects; reader reconstructing a stock-equivalent
  index from them.
- Fork DiskANN to read tier-separated blocks instead of co-located ones (§2 — the big one).
- Mutation logger + exact-reconstruction harness.

**Exit:** an index checkpointed, torn down, fully restored from the pool, and bit-identical to
the original; logger replay bit-identical after a `SIGKILL`.

### P2 — Shim and planner (weeks 8–13)

- Tier state table; modified beam search with all gates.
- Planner: queue, gain table, dependencies, token bucket.
- Miss policy with skip/fetch/adaptive modes.
- RADOS fetcher with telemetry.

**Exit:** kill a node, watch recall climb from zero with no query stalls, and `R(t)` reproduced
end to end at 10M — the first real Figure A, at small scale.

### P3 — Estimator, injection, baselines (weeks 13–18)

- Estimator training pipeline, inline inference, calibration.
- F1, F2, F2d injection scripted.
- B0 (snapshot + logical replay), B3 (frequency warm-up, block on miss), B4 (demand-driven,
  block on miss), B5 (search-effort compensation, no reordering) as switches in the same binary.
  B5 is a search-parameter setting, not a restoration policy, so it costs one flag.

**Exit:** the full baseline matrix runs unattended at 10M and produces plottable output.

### P4 — Scale and ablations (weeks 18–23)

- 100M across SIFT, Deep, MS MARCO, ADIOS2 corpus.
- B1 (GPU rebuild on Jetstream2), B2 (Weaviate as shipped).
- Ablations A1–A6.
- 1B if and only if everything above is done and stable.

**Exit:** Figures A–D drawn from real data at 100M.

### P5 — Write-up (weeks 23–26)

Paper, artifact packaging, trace release. Target: FAST '28 spring cycle (expected ~March 2027,
to be confirmed when the call appears), PVLDB rolling as fallback.

### Cut list, in order

1. The 1B run → report 100M with the Figure D trend, stated plainly.
2. B2 (Weaviate) beyond 10M.
3. F2d degraded-pool variant.
4. Approximate-betweenness proxy.
5. The ADIOS2 corpus, if embedding generation proves slow — though this is the one the NSF
   award cares about, so cut it last and reluctantly.

Already out of scope: the HNSW system port, partial-segment corruption.

### Week-one checklist

- [ ] Write down H1/H2 thresholds and commit them to the repo, dated, before any measurement
- [ ] Build 10M SIFT and Deep indexes; verify stock recall matches published numbers
- [ ] Ablation harness skeleton: load index, mask node set, measure recall
- [ ] Precompute ground truth for all query sets
- [ ] First curve: random vs. in-degree order, codes+adj+vec all-or-nothing, 20 sample points
- [ ] Extend the harness to restore a named group set, so the cross-group test needs no new code

---

## 9. Risks specific to the build

| Risk | Signal | Response |
|---|---|---|
| Tier separation costs too much steady-state performance | Intact-index `P99` regresses >20% vs. stock | Restore co-located blocks lazily in the background once all tiers are present; measure and report the overhead |
| DiskANN fork diverges from upstream | Merge pain during P1 | Pin one upstream commit at the start and stay there; this is a research artifact, not a maintained fork |
| Estimator inference too slow | >10 µs per query | Shrink the model; the fallback global curve is already implemented |
| Ceph cluster contention with other users | Throughput variance across runs | Measure throughput per run and report variance; reserve windows where possible |
| Open-loop driver saturates before the index does | Queueing dominates the measurement | Calibrate arrival rate against intact-index capacity; report the fraction of capacity used |
| 100M ground-truth computation is slow | Blocks evaluation | Compute on Jetstream2 GPUs during P1, not when first needed |

---

## 10. What this document commits to

The things a reviewer or advisor should hold this plan to:

1. Search never blocks on a missing unit unless the miss policy chose to fetch.
2. With all tiers present, results are bit-identical to stock DiskANN.
3. Logged-and-replayed graphs are bit-identical to pre-crash graphs.
4. Every baseline is a switch in the DuraVec binary, not a separate implementation, except B1
   and B2 which are external by design.
5. The H1/H2 thresholds are written before the pilot runs.
6. The corpus where tier-first restoration looks worst (SIFT) is reported alongside the one
   where it looks best (MS MARCO).
7. The planner's gain model is validated against realized gain before the planner is built, not
   assumed to be locally separable.
