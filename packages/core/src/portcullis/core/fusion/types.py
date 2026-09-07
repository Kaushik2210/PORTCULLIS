"""Fusion types.

Deliberately dependency-light: this module does the arithmetic that combines
already-computed layer scores into one calibrated probability. It has no
opinion about how those scores were produced - no ONNX Runtime, no
sentence-transformers, no FAISS. That heavier work is a training-time
concern (packages/training/.../fusion/fit.py fits the weights this module
loads and applies) and, from Milestone 6 on, a gateway-request-time concern
neither of which this package needs to import to do its own job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class LayerScores:
    """One score per cascade layer that exists as of this milestone.

    Milestone 5 only has L0 (obfuscation_score), L1 (rule engine score), L2
    (calibrated P(attack)) and the kNN sidecar. L3-L5 are unbuilt; adding
    them later means adding a field here and refitting weights, not a
    breaking change to this type's shape for existing callers.
    """

    l0: float
    l1: float
    l2: float
    knn: float

    def as_dict(self) -> dict[str, float]:
        return {"l0": self.l0, "l1": self.l1, "l2": self.l2, "knn": self.knn}


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """A fitted logistic regression, stored as plain numbers.

    Same convention as L2's Platt scaling (calibrate.py): two floats and an
    intercept, not a pickled sklearn object, so nothing that ever *applies*
    these weights needs scikit-learn as a runtime dependency - only fitting
    them does.
    """

    intercept: float
    coef: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> FusionWeights:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(intercept=float(data["intercept"]), coef=dict(data["coef"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"intercept": self.intercept, "coef": self.coef}, indent=2),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class FusionResult:
    score: float
    contributions: dict[str, float]


class MaxBaseline(NamedTuple):
    """The max() comparison point the spec asks fusion to be measured
    against (ADR-0001 names this explicitly as the honesty check: if
    learned fusion doesn't beat max(), say so)."""

    value: float
    driver: str | None
