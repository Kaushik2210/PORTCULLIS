"""The L4 state machine's transition logic - the detection logic ADR-0008
documents, tested the way L0-L2 and fusion already are: pure functions,
explicit `now`, no real clock or model involved.
"""

from __future__ import annotations

from portcullis.core.l4 import (
    ConversationRecord,
    ConversationState,
    TurnSignal,
    advance,
    apply_floor,
)
from portcullis.core.l4.machine import DEFAULT_HALF_LIFE_S
from portcullis.core.policy import Verdict

_BENIGN = TurnSignal(score=0.02, verdict=Verdict.ALLOW, taxonomy_labels=())


def _turn(
    score: float, *, verdict: Verdict = Verdict.ALLOW, labels: tuple[str, ...] = ()
) -> TurnSignal:
    return TurnSignal(score=score, verdict=verdict, taxonomy_labels=labels)


def test_a_single_benign_turn_stays_normal() -> None:
    record = advance(ConversationRecord(), _BENIGN, now=1000.0)
    assert record.state is ConversationState.NORMAL
    assert record.turn_count == 1


def test_repeated_benign_turns_never_leave_normal() -> None:
    record = ConversationRecord()
    for i in range(10):
        record = advance(record, _BENIGN, now=1000.0 + i * 10)
    assert record.state is ConversationState.NORMAL


def test_cumulative_risk_crosses_the_probing_bar() -> None:
    record = ConversationRecord()
    # Each turn scores above the noise floor (0.05), so it actually
    # contributes; turns close together so decay barely erodes the sum.
    for i in range(4):
        record = advance(record, _turn(0.10), now=1000.0 + i)
    assert record.cumulative_risk >= 0.15
    assert record.state is ConversationState.PROBING


def test_decay_reduces_risk_across_a_long_gap() -> None:
    close = ConversationRecord()
    close = advance(close, _turn(0.2), now=0.0)
    close = advance(close, _BENIGN, now=1.0)  # 1 second later

    far = ConversationRecord()
    far = advance(far, _turn(0.2), now=0.0)
    far = advance(far, _BENIGN, now=DEFAULT_HALF_LIFE_S * 10)  # ~50 minutes later

    assert far.cumulative_risk < close.cumulative_risk


def test_decay_over_one_half_life_roughly_halves_prior_risk() -> None:
    # 0.45 - the 0.05 noise floor = 0.40 contributed; the second turn's
    # score sits exactly at the floor, so it contributes nothing new and
    # this isolates the decay math.
    record = advance(ConversationRecord(), _turn(0.45), now=0.0)
    record = advance(record, _turn(0.05), now=DEFAULT_HALF_LIFE_S)
    assert 0.19 < record.cumulative_risk < 0.21


def test_a_sustained_rise_triggers_establishing_via_crescendo() -> None:
    record = ConversationRecord()
    for score in (0.02, 0.05, 0.10, 0.20):
        record = advance(record, _turn(score), now=record.last_turn_at + 1)
    assert record.state is ConversationState.ESTABLISHING


def test_a_single_spike_after_calm_turns_is_not_a_crescendo() -> None:
    record = ConversationRecord()
    for score in (0.02, 0.02, 0.02, 0.30, 0.02):
        record = advance(record, _turn(score), now=record.last_turn_at + 1)
    # One spike, then it drops straight back down - not a sustained climb.
    assert record.state is not ConversationState.ESTABLISHING


def test_persona_then_escalation_label_triggers_role_drift() -> None:
    record = ConversationRecord()
    record = advance(record, _turn(0.1, labels=("role_play_jailbreak",)), now=0.0)
    assert record.state is ConversationState.ESTABLISHING  # persona alone already escalates
    record = advance(record, _turn(0.1, labels=("direct_override",)), now=10.0)
    assert record.state is ConversationState.EXPLOITING


def test_escalation_label_alone_without_a_prior_persona_is_not_role_drift() -> None:
    record = advance(ConversationRecord(), _turn(0.05, labels=("direct_override",)), now=0.0)
    assert record.state is not ConversationState.EXPLOITING


def test_topic_pivot_after_a_challenge_verdict_triggers_exploiting() -> None:
    record = ConversationRecord()
    record = advance(record, _turn(0.1, verdict=Verdict.CHALLENGE), now=0.0)
    pivot_turn = TurnSignal(
        score=0.02, verdict=Verdict.ALLOW, taxonomy_labels=(), topic_similarity_to_previous=0.1
    )
    record = advance(record, pivot_turn, now=5.0)
    assert record.state is ConversationState.EXPLOITING


def test_similar_followup_after_a_challenge_is_not_a_topic_pivot() -> None:
    record = ConversationRecord()
    record = advance(record, _turn(0.1, verdict=Verdict.CHALLENGE), now=0.0)
    followup = TurnSignal(
        score=0.02, verdict=Verdict.ALLOW, taxonomy_labels=(), topic_similarity_to_previous=0.9
    )
    record = advance(record, followup, now=5.0)
    assert record.state is not ConversationState.EXPLOITING


def test_apply_floor_never_lowers_a_stricter_fusion_verdict() -> None:
    # Fusion alone already said BLOCK; ESTABLISHING's floor (FLAG) must not
    # water that down.
    assert apply_floor(Verdict.BLOCK, ConversationState.ESTABLISHING) is Verdict.BLOCK


def test_apply_floor_raises_a_lenient_fusion_verdict() -> None:
    assert apply_floor(Verdict.ALLOW, ConversationState.ESTABLISHING) is Verdict.FLAG
    assert apply_floor(Verdict.ALLOW, ConversationState.EXPLOITING) is Verdict.CHALLENGE


def test_apply_floor_is_a_no_op_for_states_without_a_floor() -> None:
    assert apply_floor(Verdict.ALLOW, ConversationState.NORMAL) is Verdict.ALLOW
    assert apply_floor(Verdict.ALLOW, ConversationState.PROBING) is Verdict.ALLOW


def test_apply_floor_never_reduces_strictness_across_every_combination() -> None:
    for verdict in Verdict:
        for state in ConversationState:
            assert apply_floor(verdict, state).value >= verdict.value


def test_established_labels_accumulate_across_turns() -> None:
    record = ConversationRecord()
    record = advance(record, _turn(0.0, labels=("encoding_evasion",)), now=0.0)
    record = advance(record, _turn(0.0, labels=("indirect_injection",)), now=1.0)
    assert record.established_labels == {"encoding_evasion", "indirect_injection"}


def test_new_embedding_is_carried_through_opaquely() -> None:
    record = advance(ConversationRecord(), _BENIGN, now=0.0, new_embedding=(0.1, 0.2, 0.3))
    assert record.last_embedding == (0.1, 0.2, 0.3)


def test_embedding_is_dropped_not_preserved_when_omitted() -> None:
    record = advance(ConversationRecord(), _BENIGN, now=0.0, new_embedding=(0.1, 0.2))
    record = advance(record, _BENIGN, now=1.0)  # no new_embedding this time
    assert record.last_embedding == ()


def test_recent_scores_window_is_bounded() -> None:
    record = ConversationRecord()
    for i in range(20):
        record = advance(record, _turn(0.01 * i), now=record.last_turn_at + 1)
    assert len(record.recent_scores) <= 8
