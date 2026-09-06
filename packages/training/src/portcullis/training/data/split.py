"""Family-disjoint splitting and leakage remediation.

The disjointness rule (ADR-0004 / spec: "disjoint by attack family and
source, not by row") is implemented as two policies stacked together:

1. **Source-level roles are fixed, not learned.** PromptShield is the
   backbone: its own train/validation/test skeleton is respected. Every
   other Tier-1 source is wholly a held-out evaluation slice and never
   contributes a single row to train. hard_negatives is always eval-only.
2. **Cluster-level remediation removes what the source-level rule cannot
   see.** A near-duplicate cluster (dedup.py) spanning two roles is not
   allowed to exist in the output - one side is dropped. The backbone's
   train side is always kept; the side dropped is whichever side is "less
   trusted": test/validation lose to train, and cross_source_eval loses to
   train (a "held-out" example that turns out to be a near-copy of a
   training example is not measuring what it claims to measure).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import Row

# Priority order when a cluster spans multiple partitions: lower index wins,
# i.e. rows there are kept and every other member of the cluster is dropped.
_PARTITION_PRIORITY = ("train", "validation", "test", "cross_source_eval", "hard_negatives_eval")


@dataclass(frozen=True, slots=True)
class SplitFindings:
    cross_split_clusters_remediated: int = 0
    within_split_duplicates_collapsed: int = 0
    dropped_row_ids: tuple[str, ...] = field(default_factory=tuple)


def _target_partition(source: str, source_split: str) -> str:
    if source == "hard_negatives":
        return "hard_negatives_eval"
    if source == "promptshield":
        return {"train": "train", "validation": "validation", "test": "test"}.get(
            source_split, "test"
        )
    return "cross_source_eval"


def assign_partitions(
    rows_by_source: dict[str, list[Row]],
    clusters: dict[str, int],
) -> tuple[dict[str, list[Row]], SplitFindings]:
    """Assign every row to exactly one partition, then remediate cluster
    leakage across partitions. Deterministic: iteration order is fixed by
    source name and row id, never by dict/set ordering.
    """
    provisional: dict[str, str] = {}  # row_id -> partition
    by_id: dict[str, Row] = {}

    for source in sorted(rows_by_source):
        for row in rows_by_source[source]:
            by_id[row.id] = row
            provisional[row.id] = _target_partition(row.source, row.source_split)

    # Group provisional assignment by cluster to find cross-partition leaks
    # and same-partition duplicates in one pass.
    cluster_members: dict[int, list[str]] = {}
    for row_id in sorted(provisional):
        cid = clusters.get(row_id)
        if cid is None:
            continue
        cluster_members.setdefault(cid, []).append(row_id)

    final: dict[str, str] = dict(provisional)
    dropped: list[str] = []
    cross_split_remediated = 0
    within_split_collapsed = 0

    for _cid, members in sorted(cluster_members.items()):
        partitions_present = {provisional[m] for m in members}
        if len(partitions_present) > 1:
            winner = min(partitions_present, key=_PARTITION_PRIORITY.index)
            keep = next(m for m in members if provisional[m] == winner)
            for m in members:
                if m != keep:
                    dropped.append(m)
            cross_split_remediated += 1
        elif len(members) > 1:
            # Same partition, multiple members: keep one representative.
            keep = members[0]
            dropped.extend(members[1:])
            within_split_collapsed += 1

    for row_id in dropped:
        del final[row_id]

    partitions: dict[str, list[Row]] = {name: [] for name in _PARTITION_PRIORITY}
    for row_id in sorted(final):
        partitions[final[row_id]].append(by_id[row_id])

    findings = SplitFindings(
        cross_split_clusters_remediated=cross_split_remediated,
        within_split_duplicates_collapsed=within_split_collapsed,
        dropped_row_ids=tuple(dropped),
    )
    return partitions, findings


def partition_source_counts(partitions: dict[str, list[Row]]) -> dict[str, Counter[str]]:
    """Per-partition, per-source row counts - used by the manifest and the
    dataset card to show composition, not just totals."""
    return {name: Counter(r.source for r in rows) for name, rows in partitions.items()}
