"""The kNN sidecar's index: nearest-neighbour search over unit-normalised
attack-corpus embeddings.

Implemented as a plain numpy brute-force search, not FAISS/hnswlib, per
ADR-0003's own follow-through: at this corpus scale (~6.8k attack rows,
384-dim embeddings) a single matrix multiply against the full corpus is a
few hundred microseconds, and ADR-0003 explicitly said to measure before
assuming an ANN index is warranted rather than default to one. The
measurement is in ADR-0003's amendment; if the corpus grows by an order of
magnitude or two, that measurement - not a guess - is what should trigger
revisiting this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """One corpus row the index can point back to."""

    id: str
    text: str
    family: str


@dataclass(frozen=True, slots=True)
class Neighbour:
    entry: IndexEntry
    score: float


class BruteForceIndex:
    """Cosine similarity via dot product over pre-normalised vectors.

    `embeddings` must already be L2-normalised (embed.py's contract) - this
    class does not renormalise, so a caller that skips normalisation gets
    silently wrong scores rather than a clear error. That trade favours
    keeping the hot path a single matmul with no per-call safety check.
    """

    def __init__(self, embeddings: np.ndarray, entries: list[IndexEntry]) -> None:
        if embeddings.shape[0] != len(entries):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but {len(entries)} entries given"
            )
        self._embeddings = embeddings
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def query(self, query_embedding: np.ndarray, k: int) -> list[Neighbour]:
        if len(self._entries) == 0:
            return []
        scores = self._embeddings @ query_embedding
        k = min(k, len(self._entries))
        # argpartition is O(n) rather than a full O(n log n) sort - only the
        # top k need to be ordered, not the whole corpus.
        top_k = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        ordered = top_k[np.argsort(-scores[top_k])]
        return [Neighbour(entry=self._entries[i], score=float(scores[i])) for i in ordered]
