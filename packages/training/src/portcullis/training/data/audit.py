"""The leakage audit.

Two distinct jobs live here, and they must not be confused:

* `leakage_audit` re-derives, independently of split.py, whether the *final*
  partitions actually satisfy disjointness. It is a trust-but-verify check
  on the splitter's own output, not a restatement of the splitter's logic.
* `naive_random_split_leakage` answers the question the spec asks for
  directly: "what would the naive random-split number have been?" It pools
  everything, ignores source and cluster structure, splits randomly, and
  measures how much cross-split duplication that produces - which is what
  the family-disjoint split exists to avoid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schema import Row


@dataclass(frozen=True, slots=True)
class LeakageReport:
    leaking_clusters: tuple[int, ...] = field(default_factory=tuple)
    leaking_row_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_clean(self) -> bool:
        return not self.leaking_clusters


def leakage_audit(
    partitions: dict[str, list[Row]],
    clusters: dict[str, int],
) -> LeakageReport:
    """Does any near-duplicate cluster have members in more than one
    partition? If so, the split is leaking regardless of what produced it.
    """
    cluster_to_partitions: dict[int, set[str]] = {}
    cluster_to_rows: dict[int, list[str]] = {}

    for partition_name, rows in partitions.items():
        for row in rows:
            cid = clusters.get(row.id)
            if cid is None:
                continue
            cluster_to_partitions.setdefault(cid, set()).add(partition_name)
            cluster_to_rows.setdefault(cid, []).append(row.id)

    leaking = sorted(cid for cid, parts in cluster_to_partitions.items() if len(parts) > 1)
    leaking_rows = sorted(row_id for cid in leaking for row_id in cluster_to_rows[cid])
    return LeakageReport(leaking_clusters=tuple(leaking), leaking_row_ids=tuple(leaking_rows))


def naive_random_split_leakage(
    rows: list[Row],
    clusters: dict[str, int],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> float:
    """Pool every row regardless of source, shuffle with `seed`, split by
    `ratios`, and return the fraction of test-partition rows that share a
    near-duplicate cluster with something in the train partition.

    This is deliberately the *wrong* way to split a benchmark - it is
    reproduced here specifically to quantify how wrong, per the spec's
    requirement to report the naive-split number alongside the real one.
    """
    if not rows:
        return 0.0

    rng = random.Random(seed)  # noqa: S311 - reproducible shuffling, not cryptography
    shuffled = rows[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = {r.id for r in shuffled[:n_train]}
    test_ids = {r.id for r in shuffled[n_train + n_val :]}

    train_clusters = {clusters[i] for i in train_ids if i in clusters}
    if not test_ids:
        return 0.0

    leaked = sum(1 for i in test_ids if i in clusters and clusters[i] in train_clusters)
    return leaked / len(test_ids)
