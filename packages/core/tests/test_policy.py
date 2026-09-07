"""The policy engine: fused score -> verdict.

Five ordered verdicts (ALLOW, FLAG, SANITISE, CHALLENGE, BLOCK) over four
threshold cut-points, plus shadow mode - evaluate and log without enforcing,
which the spec calls out as how this actually gets adopted in a real
deployment rather than an afterthought.
"""

from __future__ import annotations

from portcullis.core.policy import PolicyConfig, Verdict, decide

_CONFIG = PolicyConfig(
    flag_threshold=0.2,
    sanitise_threshold=0.4,
    challenge_threshold=0.6,
    block_threshold=0.8,
)


def test_score_below_flag_threshold_is_allow() -> None:
    assert decide(0.1, _CONFIG).verdict is Verdict.ALLOW


def test_score_at_each_threshold_boundary() -> None:
    """Boundaries are inclusive on the higher verdict - a score exactly at
    a cut point should not fall through to the weaker action, since a
    fixed-FPR threshold decision (the whole point of NFR-2 calibration) is
    meaningless if scores land exactly on the boundary and get the lenient
    outcome by an off-by-one."""
    assert decide(0.2, _CONFIG).verdict is Verdict.FLAG
    assert decide(0.4, _CONFIG).verdict is Verdict.SANITISE
    assert decide(0.6, _CONFIG).verdict is Verdict.CHALLENGE
    assert decide(0.8, _CONFIG).verdict is Verdict.BLOCK


def test_score_of_one_is_block() -> None:
    assert decide(1.0, _CONFIG).verdict is Verdict.BLOCK


def test_score_of_zero_is_allow() -> None:
    assert decide(0.0, _CONFIG).verdict is Verdict.ALLOW


def test_verdicts_are_monotonic_in_score() -> None:
    """Higher score must never produce a more lenient verdict than a lower
    one - the ordering ALLOW < FLAG < SANITISE < CHALLENGE < BLOCK has to
    actually hold for every score in [0, 1], not just at the four cut points."""
    order = [Verdict.ALLOW, Verdict.FLAG, Verdict.SANITISE, Verdict.CHALLENGE, Verdict.BLOCK]
    prev_rank = -1
    for i in range(0, 101):
        score = i / 100
        verdict = decide(score, _CONFIG).verdict
        rank = order.index(verdict)
        assert rank >= prev_rank, f"score={score}: {verdict} regressed below prior verdict"
        prev_rank = rank


def test_shadow_mode_computes_but_does_not_enforce() -> None:
    result = decide(0.9, _CONFIG, shadow=True)
    assert result.verdict is Verdict.BLOCK  # still computed accurately
    assert result.enforced is False  # but the caller must not act on it


def test_enforce_mode_is_enforced_by_default() -> None:
    result = decide(0.9, _CONFIG)
    assert result.enforced is True


def test_result_names_the_threshold_that_fired() -> None:
    """Explainability requirement: the policy decision must say which
    threshold it crossed, not just the final verdict."""
    result = decide(0.65, _CONFIG)
    assert "challenge" in result.rationale.lower()


def test_thresholds_must_be_strictly_increasing() -> None:
    import pytest

    with pytest.raises(ValueError, match="increasing"):
        PolicyConfig(
            flag_threshold=0.5, sanitise_threshold=0.3, challenge_threshold=0.6, block_threshold=0.9
        )
