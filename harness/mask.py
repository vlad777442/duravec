"""Named-group-set masking (HARNESS_JOBS.md 5.1) and the analytical byte
model (5.2).

Byte-cost convention: ADJ bytes are the real per-node stored content (4B
degree field + actual_degree*4B neighbor ids), not the reserved R*4B slot
that disk.index pads every node to. Both conventions are computed and
carried in every mask's byte breakdown (see mask_bytes); "actual" is the
default used to build restoration curves, per pilot decision 2026-09-16:
disk.index's own sector padding (~15% on MS MARCO) plus the reserved-vs-
actual degree gap (mean degree 58.8 vs R=100 on MS MARCO) are both bytes
DuraVec's tier-separated checkpoints would never move, so they must not
appear on the H1 byte axis, on either side of the ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable

import numpy as np

TIERS = ("CODES", "ADJ", "VEC")
PQ_CODE_BYTES_PER_VECTOR = 32  # fixed --QD 32 across all three corpora


@dataclass(frozen=True)
class Grouping:
    """Assigns every node to a group id. Group ids are ordered by restore
    priority: group 0 is restored first under a fraction sweep."""

    name: str
    order_description: str
    group_of_node: np.ndarray  # int64[npts]
    n_groups: int
    group_size: int


def make_rank_grouping(name: str, order: np.ndarray, group_size: int, order_description: str) -> Grouping:
    """order: node ids listed in restore-priority order (order[0] is
    restored first). Group membership is arithmetic in rank space per
    HARNESS_JOBS.md 6: group_id = position_in_order // group_size."""
    order = np.asarray(order)
    npts = order.shape[0]
    if not np.array_equal(np.sort(order), np.arange(npts)):
        raise ValueError("order must be a permutation of 0..npts-1")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    positions = np.arange(npts, dtype=np.int64)
    group_ids = positions // group_size
    group_of_node = np.empty(npts, dtype=np.int64)
    group_of_node[order] = group_ids
    n_groups = int(group_ids[-1]) + 1 if npts > 0 else 0
    return Grouping(
        name=name, order_description=order_description,
        group_of_node=group_of_node, n_groups=n_groups, group_size=group_size,
    )


@dataclass(frozen=True)
class TierByteStats:
    """Per-group aggregate byte costs for one Grouping, computed once and
    reused across an entire sweep (O(n_groups) per mask afterwards, not
    O(npts))."""

    codes_bytes: np.ndarray        # per group: count * 32B
    adj_bytes_actual: np.ndarray   # per group: sum(4 + degree*4)  [default]
    adj_bytes_reserved: np.ndarray  # per group: sum(4 + R*4)  [sector-padded convention]
    vec_bytes: np.ndarray          # per group: count * ndims * itemsize


def compute_group_byte_stats(grouping: Grouping, disk_index, pq_code_bytes=PQ_CODE_BYTES_PER_VECTOR) -> TierByteStats:
    n_groups = grouping.n_groups
    degrees = disk_index.get_all_degrees()  # vectorized, one pass
    if degrees.shape[0] != grouping.group_of_node.shape[0]:
        raise ValueError("grouping and disk_index disagree on npts")

    group_counts = np.bincount(grouping.group_of_node, minlength=n_groups)
    adj_actual_per_node = 4 + degrees.astype(np.int64) * 4
    adj_actual = np.bincount(grouping.group_of_node, weights=adj_actual_per_node, minlength=n_groups)
    adj_reserved_per_node = np.full_like(degrees, 4 + disk_index.R * 4, dtype=np.int64)
    adj_reserved = np.bincount(grouping.group_of_node, weights=adj_reserved_per_node, minlength=n_groups)

    codes = group_counts.astype(np.float64) * pq_code_bytes
    vec_bytes_per_node = disk_index.meta.ndims * disk_index.vector_dtype.itemsize
    vec = group_counts.astype(np.float64) * vec_bytes_per_node

    return TierByteStats(
        codes_bytes=codes, adj_bytes_actual=adj_actual,
        adj_bytes_reserved=adj_reserved, vec_bytes=vec,
    )


@dataclass(frozen=True)
class Mask:
    """The masking primitive (5.1): restore exactly this named set of
    groups, at these tiers. Fraction-based sweeps are callers on top."""

    grouping_name: str
    restored_groups: Dict[str, FrozenSet[int]] = field(default_factory=dict)

    def restored_group_bool(self, tier: str, n_groups: int) -> np.ndarray:
        arr = np.zeros(n_groups, dtype=bool)
        groups = self.restored_groups.get(tier)
        if groups:
            arr[np.fromiter(groups, dtype=np.int64)] = True
        return arr

    def node_predicates(self, grouping: Grouping) -> Dict[str, np.ndarray]:
        """tier -> bool[npts], whether each node has that tier restored."""
        if grouping.name != self.grouping_name:
            raise ValueError(
                f"mask was built against grouping {self.grouping_name!r}, "
                f"not {grouping.name!r}"
            )
        preds = {}
        for tier in TIERS:
            group_bool = self.restored_group_bool(tier, grouping.n_groups)
            preds[tier] = group_bool[grouping.group_of_node]
        return preds


def mask_from_groups(grouping: Grouping, groups_by_tier: Dict[str, Iterable[int]]) -> Mask:
    restored = {t: frozenset(groups_by_tier.get(t, ())) for t in TIERS}
    return Mask(grouping_name=grouping.name, restored_groups=restored)


def mask_from_fraction(grouping: Grouping, fraction: float, tiers: Iterable[str] = TIERS) -> Mask:
    """Restore the first ceil(fraction * n_groups) groups (by restore-
    priority group id) uniformly across the given tiers. fraction=1.0
    restores everything; fraction=0.0 restores nothing."""
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n_restore = int(np.ceil(fraction * grouping.n_groups))
    restored = frozenset(range(n_restore))
    return Mask(grouping_name=grouping.name, restored_groups={t: restored for t in tiers})


def mask_bytes(mask: Mask, byte_stats: TierByteStats, adj_convention: str = "actual") -> Dict[str, float]:
    """Byte breakdown for a mask. Returns per-tier bytes under the chosen
    ADJ convention (default "actual"), the total, and the ADJ figure under
    the other convention for the record (5.5 wants both captured)."""
    if adj_convention not in ("actual", "reserved"):
        raise ValueError(adj_convention)
    adj_primary = byte_stats.adj_bytes_actual if adj_convention == "actual" else byte_stats.adj_bytes_reserved
    adj_other = byte_stats.adj_bytes_reserved if adj_convention == "actual" else byte_stats.adj_bytes_actual
    per_tier_arrays = {"CODES": byte_stats.codes_bytes, "ADJ": adj_primary, "VEC": byte_stats.vec_bytes}

    def _sum(arr, groups):
        if not groups:
            return 0.0
        idx = np.fromiter(groups, dtype=np.int64)
        return float(arr[idx].sum())

    result = {tier: _sum(per_tier_arrays[tier], mask.restored_groups.get(tier)) for tier in TIERS}
    result["total"] = sum(result[t] for t in TIERS)
    result["adj_convention"] = adj_convention
    result["ADJ_other_convention"] = _sum(adj_other, mask.restored_groups.get("ADJ"))
    return result
