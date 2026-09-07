"""Sentence embeddings for the kNN sidecar.

Reuses the same base checkpoint as L2 (sentence-transformers/all-MiniLM-L6-v2,
ADR-0005) rather than introducing a second model - it is already a
general-purpose sentence-embedding model (that is what the sentence-
transformers prefix means: a base encoder plus a trained pooling head for
semantic similarity), so no second download or extra latency budget is
spent standing up a dedicated embedding model.

Built for offline/training-time use (embedding the corpus once to build the
index), but `load_embedder`/`embed_texts` are the same two calls the gateway
now reuses live: once for the kNN sidecar's query-time score (M6), and
again for L4's topic-pivot-after-refusal similarity (M7, ADR-0008) - both
share the loaded embedder rather than each standing up their own.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_CHECKPOINT)


def embed_texts(
    model: SentenceTransformer, texts: list[str], *, batch_size: int = 64
) -> np.ndarray:
    """L2-normalised embeddings (unit vectors), shape (len(texts), 384).

    Normalising here rather than at query time means every downstream
    consumer - brute-force cosine via dot product, or FAISS's inner-product
    index - gets the same convention for free and cannot silently compare
    normalised queries against un-normalised corpus vectors or vice versa.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)
