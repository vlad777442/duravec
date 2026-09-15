# DuraVec pilot — pre-registered hypotheses and thresholds

**Written: September 15, 2026. Before any pilot measurement was run.**
Vladislav Esaulov · advised by Lipeng Wan · Georgia State University

This file fixes the go/no-go criteria for the DuraVec pilot in advance. It exists so the
decision to proceed cannot be fitted to the data after seeing it. **Nothing below is to be
edited after the first restoration curve is measured.** If a definition turns out to be
ambiguous or unmeasurable as written, the resolution is appended to §6 with its own date and
reason — never a silent edit to §2 or §3.

---

## 1. What is being tested

DuraVec claims that recall in a graph-based ANN index is recoverable non-uniformly, and that
exploiting this beats what deployed systems already do. Both halves are testable offline,
before any system is built. If either fails, the project changes or stops.

---

## 2. H1 — concentration

**Statement.** Some restoration order over nodes and data tiers reaches a recall target having
restored substantially fewer bytes than order-agnostic restoration needs.

**Threshold.** H1 holds if the best structure- or tier-aware order reaches **0.9 normalized
recall@10** having restored **at most one third (33.3%)** of the bytes that the better of
sequential and random order requires to reach the same recall.

**Operational definitions.**

- *Normalized recall@10* is recall@10 of the partially restored index divided by recall@10 of
  the intact index on the same query set at the same search parameters. The intact index is
  measured once per corpus and held fixed.
- *Bytes restored* counts all tiers actually made available — codes, adjacency, and
  full-precision vectors — not node count and not object count.
- *Sequential order* is restoration in original node-ID order. *Random order* is a uniformly
  random permutation of restore units, averaged over at least 5 seeds.
- *Best structure- or tier-aware order* is the best of the importance orders and tier-first
  orders evaluated in the pilot. It may be chosen post hoc from among those; the comparison
  baseline may not.
- Beam width and all other search parameters are held at the intact index's configuration for
  the H1 test. Search-effort compensation is measured separately (§4) and is not part of H1.
- Recall is measured on the held-out query set, against ground truth computed offline on the
  full corpus.

**Corpora.** H1 must hold on **at least two of** SIFT 10M, Deep 10M, and MS MARCO. Passing on
one corpus only is a partial result and is treated as failure for go/no-go purposes, though it
is reported.

**If H1 fails:** the project stops. There is no ordering to exploit, and the second and third
contributions have no basis.

---

## 3. H2 — exploitability

**Statement.** Steady-state access frequency is not the right objective for recovery:
recall-first restoration with non-blocking serving reaches a recall-and-latency SLO
substantially sooner than what deployed systems do.

**Threshold.** H2 holds if simulated DuraVec reaches **TTRS(0.9, 2×P99)** at least **twice as
fast** (≤ 50% of the wall-clock time) as the better of B3 and B4, at the measured 100M-scale
rehydration throughput.

**Operational definitions.**

- *TTRS(r, L)* is the earliest time t after failure such that windowed normalized recall@10 is
  ≥ r and windowed P99 latency is ≤ L for all t' ≥ t. Window Δ = 1 s.
- *P99* in the bound is the intact index's 99th-percentile latency, measured once per corpus.
- A query exceeding **10× the intact index's P99** is recorded as recall 0 for its window. It is
  not dropped, and it is not excused by eventually returning.
- *B3* is frequency-ordered warm-up that blocks on a miss. *B4* is demand-driven restore that
  blocks on a miss. Both are simulated on the same curves, byte costs, and measured fetch
  latencies as DuraVec.
- *Simulated DuraVec* uses the greedy recall-per-byte planner with skip-on-miss serving. It does
  not assume the adaptive miss policy, the per-query estimator, or any correction not yet
  measured.
- The comparison is at the **measured** Ceph throughput and fetch latency from pilot step 3, not
  at an assumed or best-case rate.

**Failure mode F2 (serving-node loss, healthy pool)** is the configuration for this test.

**If H2 fails while H1 holds:** narrow to the characterization and metric as a short paper
(HotStorage-scale), and take VALOR or BoundedVec as paper three. Do not build the system.

---

## 4. Measurements that are NOT gates

These change the design but do not decide whether the project proceeds. Recording them here so
that a disappointing result cannot later be reframed as a gate, and a good one cannot be
promoted into evidence for H1 or H2.

- **Cross-group dependency.** Whether the local gain model overvalues isolated hub and bridge
  groups. Outcome selects a connectivity discount, a closure rule, or neither.
- **Bandwidth versus latency.** Whether rehydration is bandwidth-bound at our object sizes. If
  it is latency- or IOPS-bound, the planner's objective becomes recall per fetch rather than per
  byte. This changes the formulation, not the go/no-go.
- **Search-effort compensation (B5).** How much of the recall deficit a wider beam recovers
  without moving bytes, with the deficit decomposed into *absence* (true neighbor not restored;
  unrecoverable by search) and *broken navigation* (true neighbor restored but unreachable;
  recoverable). Absence-limited recall is computed exactly per query, not inferred from the gap.
  A strong B5 narrows DuraVec's margin and must be reported; it does not by itself fail H2,
  which is measured against B3 and B4.

---

## 5. Commitments

1. Thresholds in §2 and §3 are final as of the date above.
2. The corpus where tier-first restoration looks worst is reported alongside the one where it
   looks best.
3. Random-order baselines are averaged over ≥5 seeds; single-seed comparisons are not reported.
4. If a threshold is missed narrowly, it is missed. "Close to one third" is not one third, and
   the §2/§3 decision rules apply as written.
5. Null and partial results are reported in the proposal defense regardless of outcome.

---

## 6. Post-hoc clarifications

*Append only. Each entry dated, with the ambiguity it resolves and why the resolution could not
have been chosen to favor a particular outcome. No entries yet.*
