"""fit_fusion_variant on synthetic data - the ablation mechanism (ADR-0010
Decision 4) applied via the real `core.fusion.fuse()`, not a parallel
implementation of the sigmoid math.
"""

from __future__ import annotations

import numpy as np

from portcullis.core.fusion import FusionWeights, LayerScores, fuse
from portcullis.eval.fusion_variants import fit_fusion_variant


def _synthetic(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """l0 is the only informative feature; l1/l2/knn are pure noise -
    excluding l0 should visibly hurt fit quality on this data."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    l0 = labels * 0.8 + rng.normal(0, 0.05, size=n)
    noise = rng.normal(0, 0.05, size=(n, 3))
    features = np.column_stack([l0, noise])
    return features, labels


def test_variant_weights_omit_the_excluded_feature() -> None:
    features, labels = _synthetic()
    weights = fit_fusion_variant(features, labels, include=("l1", "l2", "knn"))
    assert "l0" not in weights.coef
    assert set(weights.coef) == {"l1", "l2", "knn"}


def test_excluded_feature_contributes_nothing_via_fuse() -> None:
    features, labels = _synthetic()
    weights = fit_fusion_variant(features, labels, include=("l1", "l2", "knn"))
    scores = LayerScores(l0=0.99, l1=0.1, l2=0.1, knn=0.1)  # l0 = the informative feature
    result = fuse(scores, weights)
    assert result.contributions["l0"] == 0.0


def test_removing_the_only_informative_feature_hurts_the_fit() -> None:
    features, labels = _synthetic()
    with_l0 = fit_fusion_variant(features, labels, include=("l0", "l1", "l2", "knn"))
    without_l0 = fit_fusion_variant(features, labels, include=("l1", "l2", "knn"))

    scores_pos = [
        LayerScores(l0=row[0], l1=row[1], l2=row[2], knn=row[3]) for row in features[labels == 1]
    ][:50]
    scores_neg = [
        LayerScores(l0=row[0], l1=row[1], l2=row[2], knn=row[3]) for row in features[labels == 0]
    ][:50]

    def _separation(weights: FusionWeights) -> float:
        pos = np.mean([fuse(s, weights).score for s in scores_pos])
        neg = np.mean([fuse(s, weights).score for s in scores_neg])
        return float(pos - neg)

    assert _separation(with_l0) > _separation(without_l0)
