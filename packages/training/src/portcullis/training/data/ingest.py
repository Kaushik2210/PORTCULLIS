"""Pulls each registered Tier-1 source into the common Row shape.

Only sources.TIER1_SOURCES are fetched by default (ADR-0004): everything
here is ungated, and `just data` reproduces it on a clean machine with no
account and no manual step. Tier 2 requires --include-gated and is not
exercised by this module at all - see sources.py for why.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from .schema import Row
from .sources import SourceSpec

_RESOURCES_DIR = Path(__file__).parent / "resources"


def _row_id(source: str, split: str, index: int) -> str:
    return f"{source}:{split or 'na'}:{index}"


def load_hf_source(spec: SourceSpec) -> list[Row]:
    if spec.hf_id is None:
        raise ValueError(f"{spec.name} has no hf_id; use load_local_source instead")

    rows: list[Row] = []
    ds = load_dataset(spec.hf_id)
    for split in spec.hf_splits:
        for i, example in enumerate(ds[split]):
            raw_label = example[spec.label_column]
            rows.append(
                Row(
                    id=_row_id(spec.name, split, i),
                    text=str(example[spec.text_column]),
                    label=spec.map_label(raw_label),
                    source=spec.name,
                    source_split=split,
                    license=spec.license,
                    tier=spec.tier,
                    family=spec.name,
                )
            )
    return rows


def load_local_source(spec: SourceSpec, path: Path | None = None) -> list[Row]:
    """Load an authored, in-repo JSONL corpus (currently: hard_negatives)."""
    path = path or (_RESOURCES_DIR / f"{spec.name}.jsonl")
    rows: list[Row] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            rows.append(
                Row(
                    id=_row_id(spec.name, "", i),
                    text=str(example[spec.text_column]),
                    label=spec.map_label(example[spec.label_column]),
                    source=spec.name,
                    source_split="",
                    license=spec.license,
                    tier=spec.tier,
                    family=spec.name,
                )
            )
    return rows


def load_source(spec: SourceSpec) -> list[Row]:
    return load_local_source(spec) if spec.hf_id is None else load_hf_source(spec)
