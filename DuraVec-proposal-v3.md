# DuraVec: Graded Recovery for Graph-Based Vector Indexes

**Draft research proposal — paper three · Revision 3, September 2026**
Vladislav Esaulov, advised by Lipeng Wan, Georgia State University

---

## 1. Summary

Vector indexes have become infrastructure, and at billion scale an index is hundreds of gigabytes or more of compressed codes, adjacency, and vectors that must be in place before queries are good. When a serving node fails, that state has to be reloaded or rehydrated. Existing systems do not sit idle while it happens — warm-up and lazy loading prioritize what to pull first, and demand-driven paths fetch what a query asks for. What none of them does is treat *retrieval quality* as the thing being recovered: load order is chosen by access frequency or by demand, and a query that reaches something not yet loaded waits for it.

Making vector indexes *durable* is not the open problem. Weaviate already logs link-level HNSW mutations and replays them without recomputing distances; P-HNSW (2025) makes HNSW crash-consistent on persistent memory with node and neighbor-list logs; pgvector inherits physical redo from PostgreSQL. Durability decides whether an index comes back. It says nothing about what the system does between the failure and the moment the index is whole — and for ANN indexes, that interval is where the interesting structure is.

Correctness in an ANN index is *graded*. A transactional system is consistent or it is not; an index at 90% recall is useful, and at 98% it is nearly indistinguishable from whole. Database recovery has exploited *availability* gradation — instant restore makes restored data usable before the rest arrives — but never *correctness* gradation, because transactional correctness has none. For a proximity graph, correctness gradation is unlikely to be uniform: search is funneled through a small set of hub and entry-region nodes, and navigation needs compressed codes and adjacency long before it needs full-precision vectors. If so, restoration order is an optimization problem over recall, and a partially restored graph can answer queries immediately with honestly labeled quality.

DuraVec makes that interval a design target. It proposes a metric, **time-to-recall-SLO** — the wall-clock delay after a failure before the served index again meets a recall target within a latency bound — and a recovery design that restores structure in recall-first order and serves a partial graph without stalling.

The proposal rests on two hypotheses, both testable offline before a system is built.

**H1 (concentration).** Recall is recoverable non-uniformly: some restoration order over nodes and data tiers reaches a recall target having restored a small fraction of the bytes that sequential or random restoration needs. If recall tracks restored bytes regardless of order, there is nothing to exploit and the paper dies.

**H2 (exploitability).** The question is whether steady-state access frequency is the right objective for *recovery*. Deployed strategies — access-frequency warm-up and demand-driven restore, both blocking on a miss — optimize for what queries touch, not for what recall they return. H2 holds if recall-first restoration with non-blocking serving reaches a recall-and-latency SLO substantially sooner than both. It may not: frequency and structural importance are correlated, and the experiment is designed to let frequency win. If it does, the result is a characterization, not a system.

---

## 2. Motivation

### 2.1 The cost that remains is rehydration

Rebuilding a billion-scale graph index from its corpus is the worst case. The DiskANN paper reports multi-day construction for SIFT1B on a commodity workstation, the NeurIPS '21 Big-ANN competition allowed up to four days per billion-scale build, and GPU builders (ScaleGANN; the cuVS GPU Vamana build) have cut this to hours rather than eliminated it.

Well-engineered systems rarely pay that cost after a crash. Segmented designs — Milvus, Qdrant, Lucene-based engines — replay logical operations only for data not yet flushed and load sealed per-segment indexes from storage. FreshDiskANN replays its redo log into a small in-memory index. Weaviate and pgvector replay link-level or physical records. After a *process* crash, recovery work is bounded by the checkpoint interval.

What remains expensive is losing the *serving* state. When a serving node dies, the durable copy in a replicated or erasure-coded object store survives — that is what the store is for — but a replacement node must fetch and load hundreds of gigabytes of codes, adjacency, and vectors before its index is whole, sharing I/O with live traffic and often with the storage tier's own backfill. Serving replicas mask the outage but not the cost: surviving replicas carry the full load until the replacement catches up, replicating a billion-scale serving tier multiplies its memory and flash footprint, and object-storage-backed designs deliberately place durability in the storage tier instead. The operator's question — after this node dies, how long until queries are good again? — is a question about rehydration, and we found no system or paper that answers it in terms of query quality.

### 2.2 Gradation is unexploited

Rehydration paths today are designed around completeness. Warm-up and lazy loading in shipping vector databases choose what to load first by access frequency and fetch anything missing on demand, so a query that touches unloaded structure waits. On-demand restore in databases does the same, ordered by demand. Search engines that return partial results under shard loss do skip what is missing, but at whole-shard granularity, with recovery order set by operator-assigned index priority rather than by any measure of result quality.

Two lines of evidence suggest a proximity graph offers much more. Navigable small-world graphs route most searches through a small, heavily traversed "highway" of hub nodes (Munyampirwa et al., 2024), and network science predicts that graphs tolerate random node loss until a percolation threshold while targeted loss of hubs fragments them quickly (Albert, Jeong & Barabási, 2000). Read in reverse, both predict that restoring hubs and the entry region first should recover navigability far faster than restoring in ID or random order.

A second axis is independent of topology. DiskANN-style search navigates with compressed codes and adjacency and uses full-precision vectors only to re-rank. For 768-dimensional float32 embeddings, a full vector is roughly twelve times larger than a degree-64 adjacency list, so deferring vectors defers most of the bytes while forfeiting only re-ranking accuracy.

Neither axis has been measured as a recovery property. Frequency-based warm-up exploits the first only incidentally, to cut latency on an index it treats as complete; tiered designs that keep full vectors in cold storage do so for steady-state cost, not as a recovery order.

### 2.3 Reproducibility across a failure

Graph insertion in HNSW and Vamana depends on the graph state at insertion time and is not reproducible under concurrent updates, so replaying a logical log constructs a *different* index with similar statistics. This is the standard determinism argument for physical over logical redo (ARIES; row-based over statement-based replication), not a new observation, but it matters for scientific corpora where retrieval results feed downstream analysis. DuraVec uses graph-mutation logging for exact reconstruction, and tags every result served during recovery with the restored-state version, so a pipeline can record — and later re-issue — any query answered by a partial index.

---

## 3. Recovery model

### 3.1 Index, tiers, and restore units

A graph index $G = (V, E)$ holds three tiers of per-node data: a compressed code $c_v$, an adjacency list $N(v)$, and a full-precision vector $\mathbf{x}_v$. As in DiskANN, the serving node keeps codes in memory and co-locates adjacency and vectors in fixed-size blocks on local NVMe. The durable copy is organized differently: checkpoints are written to a Ceph pool as *tier-separated* segment objects, so each tier of each node group can be fetched independently, and rehydration fills serving blocks progressively as tiers arrive.

A *restore unit* $u = (g, \text{tier})$ pairs a node group $g$ with one tier and has byte cost $b_u$. Segments are defined by node-membership lists rather than contiguous ID ranges, which lets the checkpointer apply a **hub-aware layout**: nodes are ranked by a cheap structural-importance proxy, the top-ranked nodes are packed into small priority groups, and the long tail is grouped for I/O locality. Contiguous-ID segments would mix hubs and leaves in every object and blunt any ordering; the layout is what makes node order actionable at object granularity.

### 3.2 Failure modes

- **F1 — serving-process crash.** Memory state (codes, caches, un-checkpointed updates) is lost; local NVMe survives. Recovery reloads codes and replays the log since the last checkpoint.
- **F2 — serving-node loss.** Memory and local NVMe are lost; the pool is healthy. A replacement node rehydrates every tier from the pool. This is where ordering matters most.
- **F2d — serving-node loss with a degraded pool.** As F2, but the OSDs on the failed host are down too, so rehydration reads compete with backfill in a replicated pool or require reconstruction in an erasure-coded one.

### 3.3 Metrics

Let $R(t)$ be recall@10 over queries arriving in $[t, t+\Delta)$ after the failure, normalized by the intact index's recall at the same search parameters, with ground truth computed offline for the recovered data state. Let $P_{99}(t)$ be the 99th-percentile latency over the same window. A query that exceeds a timeout (provisionally $10\times$ the intact index's $P_{99}$) counts as recall 0, so designs that stall cannot hide behind the recall of queries that eventually return.

**Time-to-recall-SLO.** For a recall target $r$ and latency bound $L$,
$$\text{TTRS}(r, L) = \min\{\, t : R(t') \ge r \ \wedge\ P_{99}(t') \le L \ \text{ for all } t' \ge t \,\}$$

**Recall deficit integral.** Over the recovery window $[0, T]$,
$$D(r) = \int_0^T \max\big(0,\; r - R(t)\big)\, dt$$

TTRS is the operator-facing number; $D$ is the planner's objective, penalizing both how far recall falls and how long it stays down. The latency term is essential: the incumbent designs trade the other way — full recall, stalled queries — and a recall-only metric would declare them perfect.

### 3.4 The planning problem

Rehydration executes a sequence of restore units under an I/O budget $B(t)$ shared with live traffic and, in F2d, with backfill. The planner chooses the sequence to minimize $D(r)$ subject to the latency bound.

The natural heuristic is greedy by estimated recall gain per byte, $\widehat{\Delta R}_u / b_u$, respecting tier dependencies (a node's adjacency cannot guide search until its neighbors' codes are present). Greedy carries no near-optimality guarantee here: recall gain is not submodular in the restored set, because percolation creates complementarities — a bridge helps only once both sides are present. The planner therefore takes gain estimates from offline ablation curves, applies limited lookahead over dependent units, and is measured against an **evaluation-greedy reference** that, on a small index, restores at each step whichever unit most increases *measured* recall. Optimal orders are intractable and are not claimed.

Importance proxies are computed at checkpoint time: in-degree, hop distance from the entry point, HNSW level, and traversal frequency over a sample query workload. The last is exactly what a warm-up policy would use; whether recall-aware ordering improves on it is part of H2, not an assumption.

### 3.5 Logging for F1 (enabling mechanism)

Two log disciplines are compared for F1. A *logical* log records $\langle \text{insert}, v, \mathbf{x}_v\rangle$ and $\langle \text{delete}, v\rangle$. A *mutation* log records the resulting adjacency deltas $\langle u, \text{add}/\text{remove}, w \rangle$, including the back-edge changes that robust pruning induces on existing nodes. With $\ell_{\text{log}}, \ell_{\text{mut}}$ bytes logged and $\tau_{\text{log}}, \tau_{\text{mut}}$ replay cost per insert, the trade is $\ell_{\text{mut}} > \ell_{\text{log}}$ against $\tau_{\text{mut}} \ll \tau_{\text{log}}$, and the mutation log additionally reconstructs the pre-crash graph exactly.

This is not claimed as a contribution. Weaviate's HNSW commit log already records link-level operations and replays them apply-only, P-HNSW logs node and neighbor-list updates, and pgvector's redo is physical. DuraVec needs the mechanism for disk-resident Vamana, where the published designs we found — FreshDiskANN's redo log among them — log logical operations, and it reports the churn crossover as a sizing result for F1.

---

## 4. Serving under partial recovery

Ordering pays only if an incompletely restored index can answer queries. The serving shim provides three things.

**Tier-aware search semantics.** A per-tier restored-set bitmap defines what search may use. A node without a code cannot be scored; a node without adjacency cannot be expanded; a node without its full vector is ranked by its code alone. Search over a partial index is search over the restored subgraph — it degrades recall but never faults.

**An adaptive miss policy.** When beam search reaches a missing unit it can *skip* it, keeping latency bounded and losing recall, or *fetch* it, keeping recall and stalling the query. DuraVec decides per query. Early in recovery, when a fetch means pulling an object from the pool, it skips; later, or when a query's predicted recall shortfall is large and its latency budget allows, it fetches and promotes the unit in the plan. With always-fetch the shim behaves like demand-driven restore and with always-skip like pure partial serving, so both incumbents are corners of the policy space.

**An honest, per-query recall estimate.** Each response carries an estimated recall@10, the restored-state version, and a degraded flag. The estimator is a small regression model trained offline on ablated indexes. Its features describe what *this* query lost — the fraction of top-ranked expansions skipped, the candidates ranked by code alone, the distance margin between the $k$-th result and the best skipped candidate, and query hardness — because a single global curve would misjudge exactly the hard queries whose neighbors are missing. Clients can accept degraded results, wait, or re-issue later; calibration error is a reported metric.

---

## 5. System design

DuraVec is a layer over a disk-resident Vamana/DiskANN index with four components.

- **Checkpointer and layout.** Writes tier-separated, hub-aware segment objects to the pool incrementally, so checkpoint cost scales with churn rather than index size and recovery can begin from mixed-age segments.
- **Mutation logger.** Group-commits adjacency deltas at the graph-mutation boundary, for F1 and exact reconstruction (§3.5).
- **Recovery planner.** Orders restore units by estimated recall gain per byte under the shared I/O budget, and accepts promotions from the serving shim.
- **Serving shim.** Tier bitmaps, the adaptive miss policy, the recall estimator, and result provenance tags (§4).

Segments live as objects in the existing research Ceph cluster. This is not incidental: it makes node loss and pool degradation real, injectable failures rather than simulated ones, and it reuses a multi-node cluster already in service rather than spending CloudLab reservation time standing one up.

---

## 6. Evaluation

### 6.1 Testbed and data

The existing multi-node research Ceph cluster is the object tier, with failures injected by stopping OSDs and hosts. CloudLab storage-dense nodes with independent NVMe devices host the serving tier, where per-device queueing must not be an artifact of partitioning one disk. Jetstream2 GPUs generate embeddings and build indexes, which keeps the rebuild baseline affordable to measure rather than estimate.

Corpora: SIFT and Deep at 10M and 100M (subsets of SIFT1B and Deep1B), with a 1B run as a stretch goal; MS MARCO passage embeddings (about 8.8M) as the high-dimensional workload where vector-tier deferral matters most, with Big-ANN Text-to-Image as a higher-dimensional option at 100M; and a scientific corpus drawn from the ADIOS2 ecosystem, which is the case NSF 2531903 cares about and the one where reproducible retrieval matters most. HNSW is measured offline in the pilot; the system targets Vamana.

Update workload for F1: streaming insert/delete churn at several rates, since churn governs the logging crossover.

### 6.2 Baselines

- **B0 — snapshot plus logical replay**, FreshDiskANN-style, on the same Vamana codebase (F1).
- **B1 — full rebuild from corpus** with a GPU builder, measured at 10M–100M and cited at 1B.
- **B2 — Weaviate as shipped** (HNSW commit log, snapshots, lazy shard loading), for external validity against the incumbent link-level logging design.
- **B3 — frequency-ordered warm-up, block on miss.** Restore by traversal frequency and fetch on demand, modeling warm-up and lazy loading in shipping systems and DiskANN's hot-node cache.
- **B4 — demand-driven restore, block on miss.** Restore what queries request, in the style of database instant restore.
- **B5 — search-effort compensation, no reordering.** Restore in sequential order but widen the beam, trading latency for recall over whatever subgraph is present. It moves no extra bytes, so it is the cheapest possible response to partial restoration and the first thing a reviewer will propose. Its ceiling is set by §6.2.1.

### 6.2.1 Two kinds of recall loss

B5 is worth running because partial restoration causes recall loss in two ways that respond to
different remedies.

**Absence.** A true neighbor's data is not present. No amount of search effort recovers it;
only restoration does. This is a hard ceiling on recall at a given restored set.

**Broken navigation.** A true neighbor *is* restored, but the paths that reached it run through
unrestored nodes, so search does not find it. A wider beam can route around the holes.

Ordering addresses both; search effort addresses only the second. Measuring B5 against DuraVec
therefore partitions the recall deficit into the part restoration order can fix and the part
search parameters can fix, which is a result in its own right. It also bounds the risk: B5 buys
recall with latency, which the TTRS bound already prices, so it cannot dominate — at worst it
narrows DuraVec's margin at high restored fractions, where absence is rare and navigation is
most of the loss. The expectation, to be tested rather than assumed, is that absence dominates
early in recovery, which is exactly the regime TTRS measures.

### 6.3 Ablations

Policy switches in the same codebase: **A1** random node order; **A2** importance order over contiguous-ID segments (no hub-aware layout); **A3** whole-record restoration (no tier separation); **A4** always-skip, always-fetch, and adaptive miss policies; **A5** global versus per-query recall estimator; **A6** the evaluation-greedy reference order, on a 1M index with coarse restore units;
**A7** DuraVec ordering with and without search-effort compensation, to test whether the two
compose or overlap.

### 6.4 Failure injection

F1 by killing the serving process mid-update; F2 by removing the serving node and starting a replacement; F2d by additionally stopping the OSDs on the failed host. Partial segment corruption is deferred to future work.

### 6.5 Metrics

TTRS at normalized recall 0.90, 0.95, and 0.99, each at $L = 2\times$ and $5\times$ the intact index's $P_{99}$; recall deficit integral; time spent above the latency bound; rehydration I/O volume and throughput; checkpoint cost and steady-state throughput overhead; log bytes per insert and replay time (F1); recall-estimate calibration error; and a bit-for-bit comparison of recovered and pre-crash adjacency.

### 6.6 Principal figures

**Figure A — the recovery curve.** $R(t)$ and $P_{99}(t)$ after failure for DuraVec, B0, B3, B4, and A3, per corpus and failure mode. The paper's claim in one plot: recall rising early under DuraVec while latency stays bounded, where the blocking designs either stall queries or wait for completeness.

**Figure B — restoration curves.** Recall against bytes restored under sequential, random, frequency, importance, tier-first, and evaluation-greedy orders, with and without hub-aware layout. The gap between curves is H1; a percolation knee under random order is expected and reported.

**Figure C — the regime map.** DuraVec's TTRS advantage against rehydration throughput and vector dimensionality, identifying where graded recovery matters (slow or contended storage, high-dimensional vectors) and where plain reload is good enough — reported, not concealed.

**Figure D — scale.** TTRS against index size from 10M to 100M (1B if reached), showing whether the advantage grows with scale.

Smaller figures report the F1 logging crossover and estimator calibration.

---

## 7. Pilot and go/no-go

Three weeks, before the full system is built. The first measurement takes days and should be in hand for the proposal defense.

1. **Restoration curves (days 1–5).** On 10M Vamana and HNSW indexes over SIFT and Deep (and MS MARCO once embeddings are generated), remove and restore structure offline under the orders of Figure B, sampling densely near the knee, and measure recall at fixed search parameters. No crash machinery, no cluster.
2. **Cross-group dependency (days 5–8).** The planner scores a restore unit from a local gain table, but a hub group's adjacency is worth nothing until enough of its neighbors' codes exist. Restore high-rank groups both in isolation and together with their neighbor closure, and compare realized recall gain against the local prediction. This measures whether the gain model systematically overvalues isolated hub and bridge groups — the units it most wants to schedule first.
3. **Rehydration throughput (week 2).** Read 100M-scale volumes of segment-sized objects from the Ceph pool — a real index is not needed to measure throughput — with the pool healthy and with one host's OSDs stopped. Record bulk throughput, single-fetch latency, and achieved parallelism separately: the whole design optimizes recall per *byte*, which is the right objective only if rehydration is bandwidth-bound. If it is latency- or IOPS-bound at these object sizes, fetch parallelism matters more than order and the planner's objective needs rethinking.
4. **Trace-driven simulation (weeks 2–3).** Combine the curves, byte costs, and measured fetch latencies to simulate $R(t)$ and $P_{99}(t)$ for DuraVec, B3, B4, and B5, and estimate TTRS without building the system. B5 needs no simulation machinery beyond a search-parameter sweep on the step-1 harness: measure recall against restored fraction at several beam widths, and decompose the deficit into absence and broken navigation (§6.2.1). If widening the beam closes most of the gap at low restored fractions, the ordering contribution is weaker than assumed and that must surface in the pilot rather than in review.
5. **Close out the prior-art check.** Read the warm-up and lazy-loading code paths in Milvus, OpenSearch, and Weaviate so B3 is modeled faithfully; check PipeANN, OdinANN, and Microsoft DiskANN code for adjacency-level logging; repeat the literature search on graded and partial recovery.

**Hypothesis tests.** The thresholds below are provisional and will be fixed in writing before step 1 runs, so the decision cannot be fitted to the data.

- **H1 holds** if the best structure- or tier-aware order reaches 0.9 normalized recall having restored at most one third of the bytes that the better of sequential and random order needs.
- **H2 holds** if simulated DuraVec reaches TTRS$(0.9,\ 2\times P_{99})$ at least twice as fast as the better of B3 and B4, at the measured 100M-scale throughput.

Step 2 and step 3 are not pass/fail gates but design inputs: they decide whether the planner needs a connectivity-aware gain correction, and whether recall-per-byte is the right objective at all. Both answers change the system before it is built rather than after.

**Decision rule.** H1 and H2 hold: proceed as written. H1 holds but H2 fails, meaning warm-up already captures the gradation: narrow to the characterization and metric as a short paper, and take VALOR or BoundedVec as paper three. H1 fails: kill it, having spent three weeks and almost no cluster time. Logging measurements are no longer part of the go/no-go, because they no longer decide the paper.

---

## 8. Contributions

One idea, and the machinery it requires: **ANN recovery should optimize retrieval quality, not restoration completeness.**

**Primary.**

1. **A graded recovery model for ANN indexes,** with the first measurement of how recall recovers as graph structure and data tiers are restored — under sequential, random, frequency, and importance orders, with and without hub-aware layout, across failure modes — and curves and traces released.
2. **Recall-first rehydration.** A planner over node-group and tier restore units that minimizes recall deficit under an I/O budget shared with live traffic and storage backfill.
3. **Serving under partial recovery.** Non-blocking search over a partially restored graph, with an adaptive per-query fetch-or-skip policy and provenance tags that make degraded results reproducible.

**Enabling, and claimed as such.**

4. **A tier-separated, hub-aware checkpoint layout,** which is the physical change that makes independent restoration of codes, adjacency, and vectors possible at all. Stock DiskANN co-locates vectors and adjacency in one block; without separating them there is nothing to order.
5. **Per-query recall estimation under partial restoration,** used to arbitrate fetch-versus-skip and to label degraded results. Recall prediction over an *intact* index is established work (DARTH, Quake, learned early termination); the contribution here is conditioning it on what a query had to skip, and it is positioned as a mechanism the system needs rather than a result.

**Not claimed.** The graph-mutation logger for Vamana is an enabling mechanism, positioned against Weaviate and P-HNSW.

Time-to-recall-SLO and the recall deficit integral are how the work is evaluated, not a separate contribution. Time-to-usefulness is conceptually adjacent to MTTR and to instant restore's application-facing availability goals; TTRS specializes it to approximate retrieval by adding a recall target and a latency bound. The contribution is the system that improves it.

---

## 9. Related work and delta

**Durability for vector indexes — not the gap.** Weaviate logs link-level HNSW operations (adding and replacing links per layer, tombstones, entry-point changes), replays them apply-only at startup, and adds snapshots and log condensing to bound that replay. P-HNSW (*Applied Sciences*, 2025) makes HNSW crash-consistent on persistent memory with a node log and a neighbor-list log. pgvector's HNSW receives physical redo from PostgreSQL's WAL. FreshDiskANN (§5.6) pairs a logical redo log with snapshots and replays only its in-memory temporary index. Qdrant, Milvus, and Lucene-based engines log operations and persist or seal per-segment indexes. GaussDB-Vector (PVLDB '25), Manu (PVLDB '22), Milvus (SIGMOD '21), and SingleStore-V (PVLDB '24) treat durability as a system sub-feature. DuraVec's delta: it claims no logging discipline; its contribution begins where these designs stop, between failure and full restoration.

**Restore order and miss handling — where DuraVec sits.** Prior designs occupy distinct points in a two-axis space.

| Restore order | Block on miss | Skip on miss |
|---|---|---|
| None — reload to completeness | Snapshot plus replay; full reload before serving | Partial results under shard loss (harvest/yield), whole-shard granularity |
| Demand-driven | Instant restore (Sauer, Graefe & Härder) | — |
| Access frequency | Warm-up and lazy loading in shipping vector databases; DiskANN hot-node cache | — |
| Recall-aware, over nodes × tiers | DuraVec, when a query's latency budget allows | **DuraVec**, by default early in recovery |

Instant restore is the closest intellectual ancestor: it restores on demand so that data is usable before restoration completes, but because transactional correctness is binary, order affects only *when* data is available. Harvest and yield (Fox & Brewer, HotOS '99) name the trade DuraVec exploits — answering from less than all the data — but systems apply it to whole shards with no quality-driven order. DuraVec's delta is the combination: order chosen by recall contribution at node and tier granularity, with skip-on-miss serving and per-query honesty.

**Why gradation should exist.** Munyampirwa et al. show that navigable small-world graphs route searches through a small, heavily traversed hub highway; Albert, Jeong & Barabási show that graphs tolerate random node loss far better than targeted hub loss. Graph reordering (Coleman et al., NeurIPS '22) and Starling (SIGMOD '24) show that node placement can be optimized for search. DuraVec does not claim the hub phenomenon; its delta is measuring gradation as a *recovery* property under realistic object layouts and byte costs, and exploiting it.

**Recall-aware ANN execution.** Learned adaptive early termination (Li et al., SIGMOD '20), DARTH (SIGMOD '25), Steiner-hardness (VLDB '24), and Quake (OSDI '25) all ask a version of the same question: given a complete index, how much search is needed to hit a recall target? They adapt search parameters, termination, or partition selection to do it. DARTH is the sharpest case and the closest to DuraVec's estimator: it predicts a query's current recall mid-search from search-progress and neighbor-distance features, and terminates as soon as a declared target is met. The premise is that the target is *attainable* — a query left to run reaches it — so a neighbor not yet found is one more step away. Under partial restoration a neighbor in an unrestored group is unreachable at any search effort, the attainable ceiling sits below 1 and rises as restoration proceeds, and the decision is not when to stop but what to fetch or skip. The two are complementary rather than competing: early termination bounds the effort a query spends, restoration ordering determines what that effort can find, and DARTH's approach generalizes to graph-based methods including Vamana. DuraVec asks a different question — given an index whose physical state is incomplete because of a failure, what should be restored or fetched to reach a recall-and-latency SLO soonest — and treats recall as the objective of *recovery* rather than of search. The estimator is where the two lines touch, and it is claimed only as the adaptation it is (§8).

**Update-path and SSD graph systems.** SPFresh (SOSP '23), OdinANN (FAST '26), PipeANN (Guo & Lu, OSDI '25), and Quake (OSDI '25) optimize updates, search I/O, and adaptivity; recovery is outside their scope. DiskANN and GPU builders (CAGRA, ScaleGANN, cuVS) set the rebuild baseline.

**Metrics.** MTTR measures time to completeness. The NeurIPS '23 Big-ANN streaming track averages recall over an update runbook, measuring quality under churn rather than after failure. TTRS measures time to usefulness.

**Progressive and accuracy-aware data.** MGARD, pMGARD, and RAPIDS (HPDC '23) establish that data can be decomposed so that partial availability yields bounded partial usefulness. DuraVec applies the principle to a learned graph structure, where the decomposition is discovered from topology rather than given by an error theory.

*Citation status.* Checked against primary sources, official documentation, or code in the September 2026 prior-art review: Weaviate's HNSW commit log and WAL documentation, P-HNSW, FreshDiskANN §5.6, GaussDB-Vector, OdinANN, Quake, PipeANN, ScaleGANN, the hub-highway paper, the Big-ANN streaming metric, DARTH (Chatzakis, Papakonstantinou & Palpanas, *Proc. ACM Manag. Data* 3(4), SIGMOD, Article 242, September 2025), and the FAST '27 dates. Still to verify before submission: DiskANN's SIFT1B build time and index size against the NeurIPS '19 paper; the NeurIPS '21 Big-ANN build limit; the recovery descriptions in Manu, Milvus, SingleStore-V, and SPFresh; RAPIDS; Steiner-hardness; Sauer et al. (ADBIS '17); current warm-up and lazy-loading behavior in Milvus and OpenSearch; and the release that introduced Weaviate's HNSW snapshots.

---

## 10. Venue and schedule

FAST '27's fall deadline (September 15, 2026) is out of reach. FAST '27 has two review cycles: spring (submissions March 17, 2026; notification June 4) and fall (submissions September 15; notification December 8). If FAST '28 repeats the pattern, its spring cycle would close in mid-March 2027 and notify in early June 2027. That matches the end of this plan and precedes a summer 2027 defense, so it is the primary target, to be confirmed when the FAST '28 call appears. It also leaves essentially no slack, which is why the cut list below matters.

The PVLDB rolling track, with monthly deadlines, is the fallback if the system slips by a month or more, and it suits the vector-DBMS audience. A one-shot-revision outcome at FAST would push final acceptance past the defense; whether that is acceptable depends on the pending paper-two decision, so the venue choice should be revisited once that decision arrives. If the pilot lands in the narrow outcome, the characterization fits HotStorage '27, which should not overlap with a concurrent full-venue submission of the same results.

| Weeks | Dates | Work |
|---|---|---|
| 0–3 | Sep 14 – Oct 4, 2026 | Pilot: restoration curves first (for the defense), Ceph rehydration throughput, trace-driven simulation, prior-art close-out; go/no-go |
| 3–8 | Oct 5 – Nov 8 | Tier-separated, hub-aware checkpointer; Vamana mutation logger; exact-reconstruction harness |
| 8–13 | Nov 9 – Dec 13 | Recovery planner; serving shim with tier bitmaps and adaptive miss policy |
| 13–18 | Dec 14 – Jan 17, 2027 | Ceph segment backend; recall estimator; F1, F2, and F2d injection; baselines B0, B3, B4 |
| 18–23 | Jan 18 – Feb 21 | 100M evaluation across corpora; B1 and B2; ablations A1–A6; 1B run if feasible |
| 23–26 | Feb 22 – Mar 14 | Writing, artifact packaging, trace release; submission |

If the schedule slips, cut in this order: the 1B run (report to 100M with the Figure D trend, and say so plainly); B2 beyond 10M; the F2d degraded-pool variant. The HNSW system port is already out of scope.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| "This is warm-up with a new name" | B3 and B4 are first-class baselines and H2 tests exactly this before the system exists; the delta is recall-aware order at node and tier granularity plus skip-on-miss serving with estimates |
| "Hub concentration is already known" | Cited, not claimed; the contribution is measuring and exploiting gradation under real layouts, tiers, and byte costs, which the hub literature does not address |
| "Graph-mutation logging already exists" | Agreed — Weaviate and P-HNSW; used here as an enabling mechanism for Vamana, not claimed |
| "A replicated pool loses nothing when a node dies" | Agreed, and that is the model: the cost is rehydrating serving state, not recovering lost data |
| Contiguous segments blunt ordering | Hub-aware layout over node-membership segments; ablation A2 measures the effect |
| Early skip-on-miss recall is too low to be useful | Tier-first restoration, adaptive per-query fetch, and per-query estimates that let clients choose to wait |
| Recall estimator is poorly calibrated | Calibration error is reported; fall back to a conservative global bound where the model is unreliable |
| GPU rebuild makes recovery a non-problem | Rebuild still takes hours at 1B, and rehydration dominates even when no rebuild is needed; Figure C maps where the advantage holds |
| 1B exceeds available resources | 100M is the primary scale and 1B a stretch goal |
| Concurrent work from active groups (the DiskANN line, PipeANN/OdinANN) | Repeat the literature and source check at the end of the pilot and again before submission |
| The pilot collides with the proposal defense | Restoration curves take days and run first; the defense presents them as preliminary results |

---

## Appendix A — Relation to NSF Award 2531903

The award (Wan, sole PI, Oct 2025 – Sep 2028, NSF OAC) concerns LLMs and vector indexing for the findability and accessibility of scientific data. DuraVec addresses an operational precondition for that agenda: accessible infrastructure has to stay useful through failures, not merely come back eventually. The scientific corpus instantiation, the reproducible-retrieval argument with result provenance, and the released recovery traces are all directly reportable, and the work needs no funding line beyond compute already available.

---

## Appendix B — Relation to prior directions

JANUS made data *movement* accuracy-aware. VALOR proposed making *protection at rest* accuracy-aware. DuraVec makes *recovery* accuracy-aware. The organizing principle across all three is that usefulness should be restored before completeness is — a defensible dissertation spine, and reviewers of the proposal defense are likely to find this instance the most convincing because its gradation is measured rather than modeled.

DuraVec and VALOR remain alternatives for paper three, not complements. VALOR's §4 value-ordered repair is the same idea applied to a different object type; running both would split one contribution across two weaker papers. DuraVec's advantages are a gap confirmed for graded recovery (though not for durability itself), a hotter area, and go/no-go tests that are cheaper and more decisive. VALOR's advantages are tighter continuity with JANUS and a formulation that generalizes beyond one data structure. The pilot in §7 makes that choice on evidence, and its first measurement should be run before the proposal defense.

---

## Appendix C — Changes from revisions 1 and 2

### From revision 2 (September 2026)

- **Contributions restructured** into three primary and two enabling, around a single idea: ANN recovery should optimize retrieval quality, not restoration completeness. The tier-separated layout and the per-query estimator are now claimed as enabling mechanisms, which is what they are, and the estimator is explicitly positioned against DARTH and Quake.
- **TTRS demoted** from a listed contribution to the evaluation framework, since time-to-usefulness is adjacent to MTTR and instant restore's availability goals. The contribution is the system that improves it.
- **H2 restated as a question** — is steady-state access frequency the right objective for recovery? — rather than as a bet that structural importance wins.
- **Pilot extended** with a cross-group dependency measurement (does the local gain model overvalue isolated hub groups?) and with separate recording of fetch latency and achieved parallelism, to test whether rehydration is bandwidth-bound at all. Neither is a pass/fail gate; both are design inputs.
- **Summary tightened** so it no longer implies shipping systems restore everything and block. They warm up and load on demand; what they do not do is order restoration by retrieval quality.
- **Baseline B5 added** (search-effort compensation with no reordering), together with §6.2.1 separating recall loss into absence and broken navigation, ablation A7, and a beam-width sweep in the pilot. Widening the beam costs no I/O and is the first alternative a reviewer proposes; it addresses broken navigation only, and measuring it partitions the deficit into what ordering can fix and what search parameters can fix.
- **DARTH verified and its boundary drawn explicitly** in §9: it predicts recall mid-search over a complete index where the target is attainable, whereas partial restoration puts some neighbors out of reach at any effort. The estimator's feature set now builds on DARTH's rather than paralleling it.

### From revision 1

- **Claim 1 demoted.** Graph-mutation logging ships in Weaviate (link-level HNSW commit log, confirmed in source) and is published as P-HNSW (2025); pgvector gets physical redo from PostgreSQL. The logger remains as an enabling mechanism for Vamana, and the paper now rests on graded recovery.
- **Failure model.** Node loss in a replicated or erasure-coded pool loses no data, so the modeled cost is rehydrating serving state. Added a degraded-pool variant; deferred partial-segment corruption.
- **Restore units.** Node groups × tiers (codes, adjacency, vectors), with a hub-aware, tier-separated checkpoint layout replacing contiguous-ID segments.
- **Serving.** Added an adaptive fetch-or-skip policy, a per-query recall estimator, and result provenance tags.
- **Metric.** TTRS now includes a latency bound and counts timed-out queries as misses, so blocking and non-blocking designs compare fairly.
- **Baselines.** Added frequency-ordered warm-up (B3) and demand-driven restore (B4); B2 is now Weaviate; the offline "oracle" became an evaluation-greedy reference, since optimal orders are intractable.
- **Pilot.** Replaced the linearity test with pre-registered H1/H2 thresholds, sampling near the percolation knee, a Ceph throughput measurement, and a trace-driven simulation against the incumbents; logging measurements left the go/no-go.
- **Motivation.** Tempered rebuild-cost claims (GPU builds take hours) and acknowledged that segmented systems replay only unflushed data. OrchANN was dropped from the construction-cost argument, since its focus is out-of-core search.
- **Related work.** Added instant restore, harvest/yield, the hub-highway result, percolation, recall estimation, and a design-space table.
- **Scope and venue.** 100M is the primary scale with 1B as a stretch; HNSW is measured offline only. FAST '27's fall deadline is infeasible; the target is FAST '28's spring cycle, assuming FAST keeps two cycles, with PVLDB rolling as the fallback.
