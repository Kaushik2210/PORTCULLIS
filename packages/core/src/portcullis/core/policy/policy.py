"""The policy engine: fused score -> enforcement decision.

Five verdicts over four ordered thresholds. Shadow mode is a first-class
mode here, not an afterthought bolted on later - the spec is explicit that
shadow mode is how a detector like this actually gets adopted into a real
deployment: evaluate and log every request's would-be verdict without
touching traffic, then turn enforcement on once the false-positive rate at
the chosen threshold is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Ordered least to most severe. The order itself is load-bearing -
    monotonicity (a higher score never yields a more lenient verdict) is
    checked directly against this ordering in tests."""

    ALLOW = 0
    FLAG = 1
    SANITISE = 2
    CHALLENGE = 3
    BLOCK = 4


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Four cut points on the calibrated [0, 1] fusion score.

    Configurable per tenant and per route (spec) - this type is the unit
    that gets instantiated differently per tenant/route, not a global.
    """

    flag_threshold: float
    sanitise_threshold: float
    challenge_threshold: float
    block_threshold: float

    def __post_init__(self) -> None:
        thresholds = (
            self.flag_threshold,
            self.sanitise_threshold,
            self.challenge_threshold,
            self.block_threshold,
        )
        if list(thresholds) != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
            raise ValueError(
                f"policy thresholds must be strictly increasing "
                f"(flag < sanitise < challenge < block), got {thresholds}"
            )


@dataclass(frozen=True, slots=True)
class PolicyResult:
    verdict: Verdict
    enforced: bool
    rationale: str


def decide(score: float, config: PolicyConfig, *, shadow: bool = False) -> PolicyResult:
    """Map a calibrated fusion score to a verdict.

    Boundaries are inclusive on the *higher* verdict: a score exactly at a
    threshold gets the stricter action, not the more lenient one. A
    fixed-FPR operating point (NFR-2) is meaningless if scores landing
    exactly on the boundary fall through to the softer outcome.
    """
    if score >= config.block_threshold:
        verdict, name, threshold = Verdict.BLOCK, "block", config.block_threshold
    elif score >= config.challenge_threshold:
        verdict, name, threshold = Verdict.CHALLENGE, "challenge", config.challenge_threshold
    elif score >= config.sanitise_threshold:
        verdict, name, threshold = Verdict.SANITISE, "sanitise", config.sanitise_threshold
    elif score >= config.flag_threshold:
        verdict, name, threshold = Verdict.FLAG, "flag", config.flag_threshold
    else:
        verdict, name, threshold = Verdict.ALLOW, "allow", None

    rationale = (
        f"score {score:.4f} is below the flag threshold ({config.flag_threshold:.4f}); allowed."
        if threshold is None
        else f"score {score:.4f} >= {name} threshold ({threshold:.4f})."
    )

    return PolicyResult(
        verdict=verdict,
        enforced=not shadow,
        rationale=rationale,
    )
