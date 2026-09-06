"""Content-hashed manifests (NFR-6: reproducibility).

A manifest hash is a promise: rebuild the pipeline from the same sources and
you get the same hash back, or something changed and you should know about
it. Hashing is order-independent by construction - row order is an artifact
of iteration, not a property of the dataset, and the hash must not depend on
incidental ordering.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import Row


def _row_content_hash(rows: list[Row]) -> str:
    """sha256 over a canonical, order-independent serialisation of `rows`."""
    lines = sorted(
        json.dumps(
            {"id": r.id, "text": r.text, "label": r.label, "source": r.source},
            sort_keys=True,
        )
        for r in rows
    )
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(partitions: dict[str, list[Row]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {"schema_version": 1, "partitions": {}}
    for name, rows in partitions.items():
        sources = Counter(r.source for r in rows)
        licenses = sorted({r.license for r in rows})
        tiers = sorted({r.tier for r in rows})
        manifest["partitions"][name] = {
            "row_count": len(rows),
            # JSON object keys are always strings; stringifying here rather
            # than leaving it to json.dump means the in-memory dict already
            # matches what gets written and read back, instead of silently
            # changing shape ({0: n} -> {"0": n}) across a round trip.
            "label_counts": {str(k): v for k, v in sorted(Counter(r.label for r in rows).items())},
            "sources": dict(sorted(sources.items())),
            "licenses": licenses,
            "tiers": tiers,
            "sha256": _row_content_hash(rows),
        }
    return manifest


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
