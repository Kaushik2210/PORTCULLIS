"""Wires L4 on top of the single-turn `DetectionPipeline` (M6): looks up a
conversation's prior state, embeds the current turn, advances the state
machine, applies the resulting verdict floor, and persists the updated
record - all opt-in, so a caller that never supplies a `conversation_id`
gets exactly M6's stateless behaviour back (ADR-0008).

Deliberately a wrapper around `DetectionPipeline`, not a change to it:
single-turn scoring and conversation tracking are different concerns with
different dependencies (L4 needs a store; L0-L2/kNN/fusion/policy don't),
and M6's existing tests and callers should not need to know L4 exists.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from sentence_transformers import SentenceTransformer

from portcullis.core.l1 import Scope
from portcullis.core.l4 import (
    ConversationRecord,
    ConversationStore,
    TurnSignal,
    advance,
    apply_floor,
)
from portcullis.training.knn.embed import embed_texts

from .pipeline import DetectionPipeline, DetectionResult

DEFAULT_CONVERSATION_TTL_S = 1800.0
"""30 minutes of inactivity before a conversation's state is forgotten -
longer than L4's own risk half-life (5 minutes, ADR-0008), so state has
already decayed most of the way to nothing before the store would evict it
anyway; this bounds memory/Redis usage, not the decay behaviour itself."""


def embed_fn_for(embedder: SentenceTransformer) -> Callable[[str], tuple[float, ...]]:
    """Builds the real, production `embed_fn` for `ConversationAwareDetector`
    from an already-loaded embedder (`DetectionPipeline.embedder`, so the
    same instance the kNN sidecar uses, not a second copy)."""

    def _embed(text: str) -> tuple[float, ...]:
        return tuple(embed_texts(embedder, [text])[0].tolist())

    return _embed


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """A plain dot product, not a full cosine formula with magnitude
    normalisation - valid because both vectors are already L2-normalised
    unit vectors (embed.py's own contract), the same shortcut the kNN
    sidecar's brute-force index already relies on (index.py)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class ConversationAwareDetector:
    def __init__(
        self,
        pipeline: DetectionPipeline,
        store: ConversationStore,
        embed_fn: Callable[[str], tuple[float, ...]],
        *,
        ttl_s: float = DEFAULT_CONVERSATION_TTL_S,
    ) -> None:
        """`embed_fn` rather than a raw `SentenceTransformer`: this class's
        own logic (state lookup, advance, floor, persist) needs nothing
        ML-specific, and taking a plain callable is what lets tests inject
        a trivial deterministic stand-in instead of loading a real model
        (see the gateway test suite's fake embed functions)."""
        self._pipeline = pipeline
        self._store = store
        self._embed_fn = embed_fn
        self._ttl_s = ttl_s

    def detect(
        self,
        text: str,
        *,
        conversation_id: str | None,
        scope: Scope = Scope.USER,
        shadow: bool = False,
    ) -> DetectionResult:
        result = self._pipeline.detect(text, scope=scope, shadow=shadow)
        if conversation_id is None:
            return result

        prior = self._store.get(conversation_id) or ConversationRecord()
        new_embedding = self._embed_fn(text)
        similarity = (
            _cosine_similarity(prior.last_embedding, new_embedding)
            if prior.last_embedding
            else None
        )
        turn = TurnSignal(
            score=result.score,
            verdict=result.verdict,
            taxonomy_labels=result.taxonomy_labels,
            topic_similarity_to_previous=similarity,
        )
        updated = advance(prior, turn, now=time.time(), new_embedding=new_embedding)
        self._store.set(conversation_id, updated, ttl_s=self._ttl_s)

        floored_verdict = apply_floor(result.verdict, updated.state)
        if floored_verdict is result.verdict:
            return replace(result, conversation_state=updated.state)

        return replace(
            result,
            verdict=floored_verdict,
            enforced=not shadow,
            rationale=(
                f"{result.rationale} Conversation state {updated.state.name} raised the floor "
                f"to {floored_verdict.name}."
            ),
            conversation_state=updated.state,
        )
