"""Fusion: combining per-layer scores into one calibrated probability.

The learned combination itself (logistic regression, fit offline against the
real corpus - packages/training/.../fusion/fit.py) is not what this file
tests; that requires real data and lives in the training package's own
tests. What belongs here, and must be correct independent of what the
weights happen to be, is: the arithmetic combining a fixed set of weights
with scores is right, weights load from disk correctly, and the max()
baseline this whole layer exists to be compared against is computed the
way the spec means it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from portcullis.core.fusion import FusionWeights, LayerScores, fuse, max_baseline


def test_fuse_matches_hand_computed_sigmoid() -> None:
    weights = FusionWeights(intercept=-1.0, coef={"l0": 0.5, "l1": 2.0, "l2": 3.0, "knn": 1.0})
    scores = LayerScores(l0=0.1, l1=0.2, l2=0.8, knn=0.3)
    result = fuse(scores, weights)

    z = -1.0 + 0.5 * 0.1 + 2.0 * 0.2 + 3.0 * 0.8 + 1.0 * 0.3
    expected = 1.0 / (1.0 + math.exp(-z))
    assert abs(result.score - expected) < 1e-9


def test_fuse_reports_per_layer_contribution() -> None:
    """The dashboard's decision inspector needs to show fusion arithmetic,
    not just the final number - each term (weight * score) must be visible."""
    weights = FusionWeights(intercept=0.0, coef={"l0": 1.0, "l1": 2.0, "l2": 3.0, "knn": 4.0})
    scores = LayerScores(l0=0.1, l1=0.1, l2=0.1, knn=0.1)
    result = fuse(scores, weights)

    assert result.contributions["l0"] == 0.1 * 1.0
    assert result.contributions["l1"] == 0.1 * 2.0
    assert result.contributions["l2"] == 0.1 * 3.0
    assert result.contributions["knn"] == 0.1 * 4.0


def test_fuse_score_is_bounded() -> None:
    weights = FusionWeights(intercept=0.0, coef={"l0": 100.0, "l1": 0.0, "l2": 0.0, "knn": 0.0})
    scores = LayerScores(l0=1.0, l1=0.0, l2=0.0, knn=0.0)
    assert 0.0 <= fuse(scores, weights).score <= 1.0


def test_max_baseline_is_the_maximum_of_the_four_scores() -> None:
    scores = LayerScores(l0=0.1, l1=0.9, l2=0.4, knn=0.2)
    assert max_baseline(scores).value == 0.9


def test_max_baseline_names_which_layer_drove_it() -> None:
    """Explainability applies to the baseline too - the comparison in the
    spec ("compare against max()") is meaningless if the baseline itself
    can't say why it fired."""
    scores = LayerScores(l0=0.1, l1=0.9, l2=0.4, knn=0.2)
    result = max_baseline(scores, with_driver=True)
    assert result.value == 0.9
    assert result.driver == "l1"


def test_weights_round_trip_through_json(tmp_path: Path) -> None:
    weights = FusionWeights(intercept=-0.5, coef={"l0": 0.1, "l1": 0.2, "l2": 0.3, "knn": 0.4})
    path = tmp_path / "fusion_weights.json"
    path.write_text(json.dumps({"intercept": weights.intercept, "coef": weights.coef}))

    loaded = FusionWeights.load(path)
    assert loaded == weights


def test_all_positive_inputs_push_above_midpoint() -> None:
    weights = FusionWeights(intercept=0.0, coef={"l0": 1.0, "l1": 1.0, "l2": 1.0, "knn": 1.0})
    scores = LayerScores(l0=0.5, l1=0.5, l2=0.5, knn=0.5)
    result = fuse(scores, weights)
    assert result.score > 0.5
