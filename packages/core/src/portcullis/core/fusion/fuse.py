"""Applies fitted fusion weights to a set of layer scores."""

from __future__ import annotations

import math

from .types import FusionResult, FusionWeights, LayerScores, MaxBaseline


def fuse(scores: LayerScores, weights: FusionWeights) -> FusionResult:
    """Logistic combination: sigmoid(intercept + sum(coef[k] * scores[k])).

    Per-layer contributions (coef[k] * scores[k], pre-sigmoid) are returned
    alongside the final score - the dashboard's decision inspector shows
    "fusion arithmetic" per the spec, which is meaningless without the terms
    that produced the sum, not just the sum itself.
    """
    as_dict = scores.as_dict()
    contributions = {k: weights.coef.get(k, 0.0) * v for k, v in as_dict.items()}
    z = weights.intercept + sum(contributions.values())
    score = 1.0 / (1.0 + math.exp(-z))
    return FusionResult(score=score, contributions=contributions)


def max_baseline(scores: LayerScores, *, with_driver: bool = False) -> MaxBaseline:
    """The hand-tuned max() this project's own ADR-0001 says learned fusion
    must be measured against, not merely assumed to beat."""
    as_dict = scores.as_dict()
    driver = max(as_dict, key=lambda k: as_dict[k]) if with_driver else None
    return MaxBaseline(value=max(as_dict.values()), driver=driver)
