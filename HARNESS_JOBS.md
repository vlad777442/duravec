# Task: DuraVec pilot — ablation harness and restoration curves

You are building the measurement harness for the DuraVec pilot and producing its first
restoration curves. This is the work the H1 go/no-go decision rests on.

**Read this whole file, plus `PREREGISTRATION.md` and `pilot/provenance.md`, before running
anything.** Then propose a plan and wait for approval.

Prerequisites already done (see `pilot/provenance.md`): SIFT10M, Deep10M and MS MARCO indexes
built with DiskANN at the pinned `cpp_main` commit; exact k=10 ground truth computed and
cross-validated; intact-index recall characterized.

---

## 1. The idea, in one paragraph

We want to know how recall degrades when parts of an index are missing, as a function of
*which* parts and *how many bytes* they account for. We do **not** need real restoration,
real failures, or real object storage to measure that. We need to **mask** node sets and
measure recall over the surviving subgraph. Simulated absence yields the same recall as real
absence, at a fraction of the engineering. Nothing you build here needs to survive into the
system.

## 2. Scope

**In scope:** a masked-search harness, importance proxies, the byte model, restoration curves,
the absence/navigation decomposition, and the beam-width sweep.

**Out of scope, do not build:** tier-separated on-disk formats, any modification to
`disk.index`, Ceph or `librados` anything, the recovery planner, the serving shim, the recall
estimator, the mutation logger. If a task seems to require one of these, stop and ask — it
almost certainly means the task can be reformulated as masking.

---

## 3. Implementation choice: Python, not a C++ fork

Implement beam search in Python/NumPy against the index files rather than patching DiskANN's
`cached_beam_search`.

Rationale: this harness is deliberately disposable, iteration speed matters more than
throughput, and a Python implementation keeps the masking logic readable and easy to verify.
The P1 fork of DiskANN is separate work.

**The correctness requirement this creates:** with nothing masked, your search must reproduce
the recall of stock `search_disk_index` at the same parameters, within a small tolerance.
Verify this before measuring anything, on all three corpora, and record the comparison. If
you cannot reproduce stock recall, the harness is wrong and every curve from it is worthless.

Reading the index: parse `disk.index` sectors directly (per-node layout is full-precision
vector, then a 4-byte degree, then neighbor IDs, padded into 4 KiB sectors; see
`pilot/provenance.md` for the arithmetic that confirms this), plus `_pq_pivots.bin` and
`_pq_compressed.bin` for the codes. Memory-map rather than loading whole files where possible.
Write the parser as its own module with its own unit tests — every later result depends on it.

---

## 4. Masking semantics

A mask is a per-tier predicate over nodes. Three tiers, matching the proposal: `CODES`, `ADJ`,
`VEC`. Search behavior under a mask:

- No `CODES` → the node cannot be scored, so it is dropped from candidate consideration
  entirely. Neighbor IDs pointing at it are filtered at expansion time.
- No `ADJ` → the node can be scored and can appear in results, but yields no neighbors. This
  is a **miss**; in the pilot the policy is always-skip.
- No `VEC` → the node is ranked by its PQ code distance alone, never re-ranked exactly. Count
  these per query.

Seeding: if the medoid lacks `CODES`, fall back to the highest-ranked node that has them.

Search over a masked index is search over the restored subgraph. It must never block, never
fault, and never silently substitute data from a masked tier.

---

## 5. Required interfaces (build these now, not later)

**5.1 Mask by named group set, not only by fraction.** The primitive is "restore exactly this
set of groups, at these tiers." Fraction-based sweeps are a caller on top of it. The
cross-group dependency test needs the named-set form and it is far cheaper to build now.

**5.2 Byte model.** Every mask has a byte cost, computed analytically from the as-built
parameters (R=100, PQ=32 B/vector, corpus element width — see the design doc §4.1 table).
Curves are plotted against **bytes restored**, never node count. Note that SIFT is uint8: 128 B
per vector, not 512.

**5.3 Absence ceiling, computed exactly.** For each query and each mask, you know from the
ground truth which of its true top-10 neighbors are in unmasked groups. Absence-limited recall
is therefore exact, not inferred:

```
absence_ceiling(q, mask) = |{ n in true_top10(q) : n has CODES under mask }| / 10
```

Report it as a curve alongside measured recall. Everything between measured recall and this
ceiling is **broken navigation** — neighbors that are present but unreachable. This
decomposition is the point of baseline B5 and must not be approximated by differencing two
separate runs.

**5.4 Beam width as a run parameter.** Every measurement takes the search parameter set as
input, so the B5 sweep is a loop over an existing interface.

**5.5 Determinism and provenance.** Every run writes a JSON record: mask spec, tier flags,
bytes restored, search parameters, seed, per-query recall, absence ceiling, miss counts,
code-only result counts, wall-clock. Curves are regenerated from these records, never from
in-memory state. Runs are resumable — a sweep that dies at point 14 of 20 should not restart
from zero.

---

## 6. Importance proxies

Computed once per index, cached to disk:

- **In-degree** — one pass over adjacency.
- **Hop distance from the medoid** — one BFS.
- **Traversal frequency** — replay a sample query workload (disjoint from the evaluation query
  set) and count node visits. This is what a warm-up policy would use, so it is the H2
  baseline and must be implemented, not approximated.

Grouping: rank nodes by proxy score, then group by rank so that group membership is arithmetic
in rank space. Make group size a parameter.

---

## 7. Measurements, in order

Run in this order. Report after each; do not batch them to the end.

1. **Harness validation.** Unmasked search reproduces stock recall on all three corpora.
2. **First curve (the H1 read).** SIFT10M, random vs. in-degree order, all three tiers restored
   together, ~20 points, random averaged over ≥5 seeds per `PREREGISTRATION.md`. Sample densely
   where the curve bends — a percolation knee is expected under random order.
3. **Extend to Deep10M and MS MARCO.** Same orders. Add traversal-frequency and hop-distance
   orders.
4. **Tier curves.** Codes-first, then codes+adjacency, then vectors. Expect a large effect on
   MS MARCO (88% of bytes are vectors) and very little on SIFT (23%, where adjacency dominates
   at 72%). If SIFT shows a large tier-separation gain, suspect the harness, not the data.
5. **Beam-width sweep (B5).** At several restored fractions and several beam widths, with the
   absence/navigation decomposition from 5.3.
6. **Cross-group dependency.** Restore high-rank groups in isolation, then with their neighbor
   closure; compare realized recall gain against what a per-decile local gain table predicts.

---

## 8. Reporting

- Curves as data files plus plots; plots are regenerated from the JSON records by a script.
- Append results to `pilot/provenance.md` as each measurement completes.
- For measurement 2, state plainly where the result falls relative to the H1 threshold in
  `PREREGISTRATION.md` (best structure- or tier-aware order reaches 0.9 normalized recall having
  restored ≤ one third of the bytes the better of sequential and random needs).

**Do not tune toward the threshold.** If the result misses, it misses — report the number.
Do not adjust group size, proxy choice, or search parameters to improve it and re-report
without saying so. If you believe a parameter choice was genuinely wrong (not merely
unflattering), say why, and let me decide.

---

## 9. Working style

- Propose the plan first; wait for approval before long sweeps.
- The index parser and the masking logic each get unit tests against a small synthetic index
  before being pointed at the real ones.
- Scripts take paths and parameters as arguments; no hard-coded paths.
- Report failures with the actual error. Never silently reduce the query set, the corpus, or
  the number of sample points to make something finish.
- If a measurement contradicts an expectation stated in this file, that is a finding — surface
  it, do not smooth it over.
