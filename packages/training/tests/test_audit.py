"""The leakage audit.

Two things are asserted here, and they are different claims:

1. The audit re-derives, independently of the splitter, that the *final*
   partitions have zero cross-split cluster leakage. This is a trust-but-
   verify check - it must not simply trust that assign_partitions() did its
   job, or a future bug there would go undetected.
2. The naive-random-split comparison produces the number the spec asks for:
   what leakage would have existed under a naive row-level split, to show
   the family-disjoint split actually bought something.
"""

from __future__ import annotations

from portcullis.training.data.audit import leakage_audit, naive_random_split_leakage
from portcullis.training.data.schema import Row


def _row(id_: str, source: str = "s") -> Row:
    return Row(
        id=id_,
        text=id_,
        label=1,
        source=source,
        source_split="",
        license="apache-2.0",
        tier=1,
        family=source,
    )


def test_leakage_audit_passes_on_clean_partitions() -> None:
    partitions = {
        "train": [_row("t1"), _row("t2")],
        "test": [_row("x1")],
    }
    clusters = {"t1": 0, "t2": 1, "x1": 2}
    report = leakage_audit(partitions, clusters)
    assert report.leaking_clusters == ()
    assert report.is_clean


def test_leakage_audit_catches_a_deliberately_broken_partition() -> None:
    """Constructs partitions that skip the splitter entirely, to prove the
    audit does not merely trust its caller."""
    partitions = {
        "train": [_row("t1")],
        "test": [_row("x1")],
    }
    clusters = {"t1": 0, "x1": 0}  # same cluster, different partitions: leak
    report = leakage_audit(partitions, clusters)
    assert not report.is_clean
    assert report.leaking_clusters == (0,)


def test_leakage_audit_reports_which_rows_leaked() -> None:
    partitions = {"train": [_row("t1")], "test": [_row("x1"), _row("x2")]}
    clusters = {"t1": 5, "x1": 5, "x2": 5}
    report = leakage_audit(partitions, clusters)
    assert set(report.leaking_row_ids) == {"t1", "x1", "x2"}


def test_naive_random_split_leakage_is_zero_with_no_duplicate_clusters() -> None:
    rows = [_row(f"r{i}") for i in range(40)]
    clusters = {r.id: i for i, r in enumerate(rows)}  # every row its own cluster
    frac = naive_random_split_leakage(rows, clusters, seed=0)
    assert frac == 0.0


def test_naive_random_split_leakage_is_positive_when_duplicates_span_the_pool() -> None:
    """20 duplicate pairs in a 40-row pool: a naive 80/20 split has no
    mechanism to keep pairs together, so some pairs will land on opposite
    sides purely by chance. Over many seeds the expected leakage is well
    above zero; this checks it is not identically zero by construction."""
    rows = []
    clusters = {}
    for i in range(20):
        a, b = f"a{i}", f"b{i}"
        rows.append(_row(a))
        rows.append(_row(b))
        clusters[a] = i
        clusters[b] = i
    fracs = [naive_random_split_leakage(rows, clusters, seed=s) for s in range(5)]
    assert any(f > 0.0 for f in fracs)


def test_naive_random_split_leakage_is_deterministic_given_a_seed() -> None:
    rows = [_row(f"r{i}") for i in range(30)]
    clusters = {rows[i].id: i // 2 for i in range(30)}  # pairs share a cluster
    a = naive_random_split_leakage(rows, clusters, seed=42)
    b = naive_random_split_leakage(rows, clusters, seed=42)
    assert a == b
