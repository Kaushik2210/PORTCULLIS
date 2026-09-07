"""Types for the L4 conversation state machine.

Everything here is a plain, frozen dataclass or enum - no wall-clock reads,
no I/O, no model calls. `advance()` (machine.py) is a pure function of a
`ConversationRecord` and a `TurnSignal`, which is what makes the state
machine as testable as L0-L2 and fusion already are (ADR-0008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from portcullis.core.policy import Verdict


class ConversationState(Enum):
    """Ordered least to most concerning, same convention as `Verdict`
    (policy.py) - the order is load-bearing for the transition table's own
    monotonicity tests, not just documentation."""

    NORMAL = 0
    PROBING = 1
    ESTABLISHING = 2
    EXPLOITING = 3


@dataclass(frozen=True, slots=True)
class TurnSignal:
    """What one turn contributes to the state machine - already-computed
    per-turn outputs from the M6 pipeline, not raw text. L4 never sees a
    prompt directly; it only sees what the earlier layers already decided
    about it (ADR-0008, Decision 2)."""

    score: float
    verdict: Verdict
    taxonomy_labels: tuple[str, ...]
    topic_similarity_to_previous: float | None = None
    """Cosine similarity between this turn's and the previous turn's text
    embedding, in [-1, 1] - `None` on the conversation's first turn, or
    whenever the caller doesn't have an embedding to offer (embedding is
    computed by the caller, e.g. the gateway's kNN embedder; `core` never
    imports a sentence-transformer, ADR-0008 Decision 2)."""


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """Everything the state machine needs to remember between turns -
    the value a `ConversationStore` (store.py) persists, keyed by
    conversation id."""

    state: ConversationState = ConversationState.NORMAL
    turn_count: int = 0
    cumulative_risk: float = 0.0
    last_turn_at: float = 0.0
    """Unix timestamp of the previous turn - 0.0 before the first turn, so
    the first turn's decay computation sees an "infinite" gap and decays
    to nothing, which is correct: there is no prior risk to decay."""
    established_labels: frozenset[str] = field(default_factory=frozenset)
    """Every taxonomy label seen across the conversation so far - how
    role-drift detects "a persona set three turns ago." """
    recent_scores: tuple[float, ...] = ()
    """A bounded window (machine.py caps this) of recent per-turn scores -
    what crescendo detection scans for a sustained rise."""
    last_verdict: Verdict = Verdict.ALLOW
    """The previous turn's verdict - what topic-pivot-after-refusal checks
    to know whether the previous turn was refusal-worthy (CHALLENGE/BLOCK)."""
    last_embedding: tuple[float, ...] = ()
    """The previous turn's text embedding, carried opaquely: `advance()`
    stores whatever the caller hands it and never reads the values inside -
    only the caller (the gateway, which already has the kNN sidecar's
    embedder loaded) computes an embedding or a similarity from it. This is
    what lets the *next* call compute `TurnSignal.topic_similarity_to_previous`
    without a second store lookup or a fragile "advance() then patch the
    embedding back in" two-step."""
