"""Builds the kNN sidecar's index from the training corpus and saves it to
disk, so fusion-fitting (and, later, a live gateway) load a precomputed
index rather than re-embedding ~6.8k rows (~160s measured) on every run.

Indexes attack rows only (label == 1) from train.jsonl - the sidecar's job
is "nearest known attack," and a benign row contributes nothing to that.
Family comes from train.weak.jsonl where available (ADR-0005's weak labels);
rows L1 could not place still index with an empty family string rather than
being dropped - a paraphrase of an unlabelled attack is still worth
retrieving as "this resembles a known attack," even without a family name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .embed import embed_texts, load_embedder
from .index import IndexEntry

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_INDEX_DIR = Path.home() / "AppData" / "Local" / "portcullis" / "knn_index"


def _load_families(data_dir: Path, partition: str) -> dict[str, str]:
    """row id -> the first weak-labelled family, or "" if L1 placed none."""
    from portcullis.training.l2.labels import TAXONOMY

    path = data_dir / f"{partition}.weak.jsonl"
    families: dict[str, str] = {}
    if not path.exists():
        return families
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hits = [
                TAXONOMY[i]
                for i, v in enumerate(row["labels"])
                if v == 1 and TAXONOMY[i] != "benign"
            ]
            if hits:
                families[str(row["id"])] = hits[0]
    return families


def build(*, data_dir: Path, out_dir: Path, partition: str = "train") -> None:
    families = _load_families(data_dir, partition)

    entries: list[IndexEntry] = []
    texts: list[str] = []
    with (data_dir / f"{partition}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["label"] != 1:
                continue
            entries.append(
                IndexEntry(id=row["id"], text=row["text"], family=families.get(row["id"], ""))
            )
            texts.append(row["text"])

    print(f"embedding {len(texts)} attack rows from {partition}", flush=True)
    model = load_embedder()
    embeddings = embed_texts(model, texts)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "entries.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"id": e.id, "text": e.text, "family": e.family}) + "\n")

    print(f"wrote {out_dir / 'embeddings.npy'} shape={embeddings.shape}", flush=True)
    print(f"wrote {out_dir / 'entries.jsonl'} ({len(entries)} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--partition", default="train")
    args = ap.parse_args()
    build(data_dir=args.data_dir, out_dir=args.out_dir, partition=args.partition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
