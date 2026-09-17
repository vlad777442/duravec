"""Per-run provenance records (HARNESS_JOBS.md 5.5): every masked-search run
over a query set writes one JSON record (mask spec, tier flags, bytes
restored, search params, seed, per-query recall, absence ceiling, miss
counts, code-only counts, wall-clock). Resumable: a sweep that dies partway
through skips runs whose record already exists rather than restarting from
zero. Curves are built from these records only, never from in-memory state.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .absence import absence_ceiling, recall_at_k
from .mask import Mask, TIERS, TierByteStats, mask_bytes
from .search import EntryPoints, PQCodes, PQTable, SearchParams, masked_search
from .diskann_format import DiskIndex


@dataclass(frozen=True)
class RunSpec:
    corpus: str
    grouping_name: str
    order_description: str
    restored_groups: Dict[str, List[int]]   # tier -> sorted group ids (JSON-stable)
    search_params: Dict[str, Optional[int]]  # L, W, k, io_limit
    adj_byte_convention: str = "actual"
    seed: Optional[int] = None               # e.g. random-order seed; None otherwise
    label: str = ""                          # free-text, e.g. "H1 random frac=0.3"

    def content_hash(self) -> str:
        """Excludes `label`: it's a free-text annotation, not part of what
        defines a run for caching purposes -- two specs that differ only in
        label describe the identical computation and must resolve to the
        same cached record."""
        fields = dataclasses.asdict(self)
        fields.pop("label", None)
        blob = json.dumps(fields, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_id(spec: RunSpec) -> str:
    return spec.content_hash()


def run_masked_sweep_point(
    spec: RunSpec,
    out_dir: str,
    mask: Mask,
    grouping,
    byte_stats: TierByteStats,
    disk_index: DiskIndex,
    pq_table: PQTable,
    pq_codes: PQCodes,
    entry_points: EntryPoints,
    queries: np.ndarray,             # (n_queries, ndims)
    true_topk_ids: np.ndarray,       # (n_queries, k)
    hop_distance_from_entry: Optional[np.ndarray] = None,
    force: bool = False,
) -> dict:
    """Run one sweep point (one mask, one query set, one search-param set)
    and write its JSON record. Returns the record (loaded from disk without
    recomputation if it already exists and force=False)."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id(spec)
    out_path = out_dir_path / f"{rid}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    node_pred = mask.node_predicates(grouping)
    params = SearchParams(**spec.search_params)

    per_query_recall = []
    per_query_ceiling = []
    per_query_miss = []
    per_query_code_only = []
    per_query_hops = []
    per_query_expanded = []
    fallback_seed_count = 0

    t0 = time.time()
    for qi in range(queries.shape[0]):
        q = queries[qi].astype(np.float32)
        result = masked_search(
            q, disk_index, pq_table, pq_codes, node_pred, params, entry_points,
            hop_distance_from_entry=hop_distance_from_entry,
        )
        per_query_recall.append(recall_at_k(result.ids, true_topk_ids[qi]))
        per_query_ceiling.append(
            float(absence_ceiling(true_topk_ids[qi: qi + 1], node_pred["CODES"])[0])
        )
        per_query_miss.append(result.miss_count)
        per_query_code_only.append(result.code_only_count)
        per_query_hops.append(result.hops)
        per_query_expanded.append(result.num_expanded)
        fallback_seed_count += int(result.used_fallback_seed)
    wall_clock_sec = time.time() - t0

    byte_breakdown = mask_bytes(mask, byte_stats, adj_convention=spec.adj_byte_convention)

    record = {
        "run_id": rid,
        "spec": dataclasses.asdict(spec),
        "bytes": byte_breakdown,
        "wall_clock_sec": wall_clock_sec,
        "n_queries": int(queries.shape[0]),
        "fallback_seed_count": fallback_seed_count,
        "mean_recall": float(np.mean(per_query_recall)),
        "mean_absence_ceiling": float(np.mean(per_query_ceiling)),
        "per_query": {
            "recall": per_query_recall,
            "absence_ceiling": per_query_ceiling,
            "miss_count": per_query_miss,
            "code_only_count": per_query_code_only,
            "hops": per_query_hops,
            "num_expanded": per_query_expanded,
        },
    }
    out_path.write_text(json.dumps(record))
    return record


def load_sweep_records(out_dir: str) -> List[dict]:
    records = []
    for p in sorted(Path(out_dir).glob("*.json")):
        records.append(json.loads(p.read_text()))
    return records
