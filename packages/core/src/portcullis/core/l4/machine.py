"""The L4 state machine: `advance()` folds one new turn into a
`ConversationRecord`, and `apply_floor()` turns the resulting state into a
verdict floor (ADR-0008, Decision 1 - a policy modifier, not a fusion
input).

Every threshold in this module is a reasoned default, not a fitted or
benchmarked value - there is no multi-turn labeled corpus to fit them
against (ADR-0008, Decision 3). They are named constants specifically so
that fact is visible at the call site, not buried in a magic number.
"""

from __future__ import annotations

from portcullis.core.policy import Verdict

from .types import ConversationRecord, ConversationState, TurnSignal

DEFAULT_HALF_LIFE_S = 300.0
"""Cumulative risk decays by half every 5 minutes of gap between turns -
a burst of turns barely decays between them; a conversation resumed after
a long pause has mostly forgotten its earlier risk."""

_CRESCENDO_WINDOW = 8
"""How many recent per-turn scores crescendo detection scans - bounded so
a long-running conversation's `ConversationRecord` doesn't grow forever."""

_CRESCENDO_MIN_RUN = 3
"""A crescendo needs at least this many consecutive *strictly increasing*
turns - each one worse than the last. Non-decreasing (allowing ties) was
tried first and rejected: it let a flat run of calm turns plus one later
spike count as a single "non-decreasing run" back to the start, which is a
one-off anomaly, not a sustained climb (that single turn is already fusion's
job to catch on its own merits)."""

_CRESCENDO_MIN_RISE = 0.15
"""...and the run has to actually go somewhere, not just tick upward by a
rounding error."""

_RISK_NOISE_FLOOR = 0.05
"""A turn only contributes `max(0, score - floor)` to cumulative risk, not
its raw score. Without this, purely benign turns (M6's own demo probe put
ordinary text around 0.02-0.03) still geometrically accumulate under a slow
decay and frequent turns - a burst of ordinary chat would otherwise drift
into PROBING on volume alone, which is not what "conversation risk" is
supposed to mean. Only turns that already show some individual signal above
the floor move the needle."""

_PROBING_RISK_BAR = 0.15
_EXPLOITING_RISK_BAR = 0.6

_TOPIC_PIVOT_SIMILARITY_THRESHOLD = 0.3
"""Cosine similarity below this, immediately after a CHALLENGE/BLOCK-worthy
turn, reads as "changed the subject right after being pushed back on"
rather than "rephrased the same request." """

# Matches the L1 rule taxonomy's label strings by convention (rules/*.yaml's
# own `labels:` field) - not a shared import, since core/l1 doesn't centralise
# the taxonomy as a type either (RuleMatch.labels is free-form tuple[str,...]).
_PERSONA_LABEL = "role_play_jailbreak"
_ESCALATION_LABELS = frozenset(
    {"direct_override", "system_prompt_extraction", "data_exfiltration", "tool_abuse"}
)

_STATE_VERDICT_FLOOR: dict[ConversationState, Verdict | None] = {
    ConversationState.NORMAL: None,
    ConversationState.PROBING: None,
    ConversationState.ESTABLISHING: Verdict.FLAG,
    ConversationState.EXPLOITING: Verdict.CHALLENGE,
}


def _detect_crescendo(scores: tuple[float, ...]) -> bool:
    if len(scores) < _CRESCENDO_MIN_RUN:
        return False
    run_start = 0
    for i in range(1, len(scores)):
        if scores[i] <= scores[i - 1]:  # a tie or a drop ends the run
            run_start = i
        run_len = i - run_start + 1
        if run_len >= _CRESCENDO_MIN_RUN and (scores[i] - scores[run_start]) >= _CRESCENDO_MIN_RISE:
            return True
    return False


def _detect_role_drift(
    prior_established_labels: frozenset[str], turn_labels: tuple[str, ...]
) -> bool:
    if _PERSONA_LABEL not in prior_established_labels:
        return False
    return any(label in _ESCALATION_LABELS for label in turn_labels)


def _next_state(
    *, decayed_risk: float, crescendo: bool, role_drift: bool, topic_pivot: bool, persona_set: bool
) -> ConversationState:
    if role_drift or topic_pivot or decayed_risk >= _EXPLOITING_RISK_BAR:
        return ConversationState.EXPLOITING
    if persona_set or crescendo:
        return ConversationState.ESTABLISHING
    if decayed_risk >= _PROBING_RISK_BAR:
        return ConversationState.PROBING
    return ConversationState.NORMAL


def advance(
    record: ConversationRecord,
    turn: TurnSignal,
    *,
    now: float,
    half_life_s: float = DEFAULT_HALF_LIFE_S,
    new_embedding: tuple[float, ...] = (),
) -> ConversationRecord:
    """Fold `turn` into `record`, producing the next `ConversationRecord`.

    Pure: no wall-clock read (`now` is a parameter), no I/O. `record`'s
    accumulated fields (`established_labels`, `recent_scores`,
    `cumulative_risk`) are what give the state machine memory across
    turns - `_next_state` itself looks at nothing but the current turn's
    computed signals, so a persona set three turns ago stays remembered
    because `established_labels` still contains it, not because of any
    explicit "don't downgrade" rule.

    `new_embedding`, if given, becomes the returned record's
    `last_embedding` - carried opaquely (see that field's docstring) so the
    caller's *next* `advance()` call can derive
    `TurnSignal.topic_similarity_to_previous` from it. Omitted, the
    previous embedding is dropped, not preserved - a caller that wants
    topic-pivot detection supplies a fresh one every turn it has one for.
    """
    elapsed_s = max(0.0, now - record.last_turn_at)
    # True half-life decay: 0.5 ** (elapsed / half_life), not exp(-elapsed /
    # half_life) - the latter is an e-folding time (decays to ~37% after
    # half_life_s), which would silently contradict this constant's name
    # and its own docstring ("decays by half every 5 minutes").
    decayed_risk = record.cumulative_risk * 0.5 ** (elapsed_s / half_life_s) + max(
        0.0, turn.score - _RISK_NOISE_FLOOR
    )

    recent_scores = (*record.recent_scores, turn.score)[-_CRESCENDO_WINDOW:]
    crescendo = _detect_crescendo(recent_scores)

    role_drift = _detect_role_drift(record.established_labels, turn.taxonomy_labels)

    topic_pivot = (
        record.last_verdict.value >= Verdict.CHALLENGE.value
        and turn.topic_similarity_to_previous is not None
        and turn.topic_similarity_to_previous < _TOPIC_PIVOT_SIMILARITY_THRESHOLD
    )

    persona_set = (
        _PERSONA_LABEL in record.established_labels or _PERSONA_LABEL in turn.taxonomy_labels
    )

    new_state = _next_state(
        decayed_risk=decayed_risk,
        crescendo=crescendo,
        role_drift=role_drift,
        topic_pivot=topic_pivot,
        persona_set=persona_set,
    )

    return ConversationRecord(
        state=new_state,
        turn_count=record.turn_count + 1,
        cumulative_risk=decayed_risk,
        last_turn_at=now,
        established_labels=record.established_labels | set(turn.taxonomy_labels),
        recent_scores=recent_scores,
        last_verdict=turn.verdict,
        last_embedding=new_embedding,
    )


def verdict_floor(state: ConversationState) -> Verdict | None:
    """The minimum verdict a request in `state` receives, regardless of
    what fusion alone said - `None` means no floor (fusion's verdict
    stands). See ADR-0008 Decision 1 for why this is a fixed table rather
    than a fitted weight."""
    return _STATE_VERDICT_FLOOR[state]


def apply_floor(verdict: Verdict, state: ConversationState) -> Verdict:
    floor = verdict_floor(state)
    if floor is None or verdict.value >= floor.value:
        return verdict
    return floor
