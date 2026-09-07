"""Loads a saved index (build_index.py) and answers "how attack-like is this
text, and what does it most resemble" queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .embed import embed_texts
from .index import BruteForceIndex, IndexEntry, Neighbour


@dataclass(frozen=True, slots=True)
class KnnScore:
    score: float
    nearest: Neighbour | None


class KnnScorer:
    def __init__(self, embedder: SentenceTransformer, index: BruteForceIndex) -> None:
        self._embedder = embedder
        self._index = index

    @classmethod
    def from_saved_index(cls, index_dir: Path, embedder: SentenceTransformer) -> KnnScorer:
        embeddings = np.load(index_dir / "embeddings.npy")
        entries: list[IndexEntry] = []
        with (index_dir / "entries.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    entries.append(IndexEntry(id=row["id"], text=row["text"], family=row["family"]))
        return cls(embedder, BruteForceIndex(embeddings, entries))

    def score(self, text: str, *, k: int = 1) -> KnnScore:
        """Cosine similarity to the nearest indexed attack, in [-1, 1] but
        in practice [0, 1] for semantically related short text - used
        directly as a fusion input score, no separate rescaling.

        Single-text convenience wrapper around score_batch. For scoring many
        texts (fusion-fitting, batch evaluation), call score_batch directly -
        embedding one text at a time through sentence-transformers pays a
        fixed per-call overhead that batching amortises away. Found the hard
        way at Milestone 5: fusion-fitting through this method one row at a
        time measured 2.1 rows/s: switching to score_batch measured >10x that.
        """
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str], *, k: int = 1) -> list[KnnScore]:
        query_embeddings = embed_texts(self._embedder, texts)
        results: list[KnnScore] = []
        for query_embedding in query_embeddings:
            neighbours = self._index.query(query_embedding, k=k)
            if not neighbours:
                results.append(KnnScore(score=0.0, nearest=None))
            else:
                results.append(KnnScore(score=max(0.0, neighbours[0].score), nearest=neighbours[0]))
        return results
