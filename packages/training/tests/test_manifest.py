"""Manifest content hashing.

Reproducibility (NFR-6) means a manifest hash is a promise: the same content
always produces the same hash, and any change to the content is detected.
"""

from __future__ import annotations

import json
from pathlib import Path

from portcullis.training.data.manifest import build_manifest, write_manifest
from portcullis.training.data.schema import Row


def _row(id_: str, source: str = "s") -> Row:
    return Row(
        id=id_,
        text=f"text for {id_}",
        label=1,
        source=source,
        source_split="train",
        license="apache-2.0",
        tier=1,
        family=source,
    )


def test_manifest_records_per_partition_counts_and_provenance() -> None:
    partitions = {"train": [_row("a"), _row("b")], "test": [_row("c")]}
    manifest = build_manifest(partitions)
    assert manifest["partitions"]["train"]["row_count"] == 2
    assert manifest["partitions"]["test"]["row_count"] == 1
    assert manifest["partitions"]["train"]["sources"] == {"s": 2}


def test_manifest_hash_is_stable_for_identical_content() -> None:
    partitions = {"train": [_row("a"), _row("b")]}
    m1 = build_manifest(partitions)
    m2 = build_manifest(partitions)
    assert m1["partitions"]["train"]["sha256"] == m2["partitions"]["train"]["sha256"]


def test_manifest_hash_changes_when_content_changes() -> None:
    m1 = build_manifest({"train": [_row("a")]})
    m2 = build_manifest({"train": [_row("a"), _row("b")]})
    assert m1["partitions"]["train"]["sha256"] != m2["partitions"]["train"]["sha256"]


def test_manifest_hash_is_row_order_independent() -> None:
    """Row order is an artifact of how the pipeline happened to iterate, not
    a property of the dataset - the hash should not depend on it."""
    m1 = build_manifest({"train": [_row("a"), _row("b")]})
    m2 = build_manifest({"train": [_row("b"), _row("a")]})
    assert m1["partitions"]["train"]["sha256"] == m2["partitions"]["train"]["sha256"]


def test_write_manifest_round_trips_through_json(tmp_path: Path) -> None:
    partitions = {"train": [_row("a")], "test": [_row("b")]}
    manifest = build_manifest(partitions)
    out = tmp_path / "corpus.manifest.json"
    write_manifest(manifest, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == manifest
