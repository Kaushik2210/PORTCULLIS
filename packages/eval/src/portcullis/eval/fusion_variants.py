"""Ablation fusion variants (ADR-0010, Decision 4): refit logistic
regression on `validation` with a feature held out entirely, so the
remaining features' own coefficients are refit to compensate - "the best
possible fusion without this signal," not "today's fusion with one input
zeroed out." Reuses `core.fusion.fuse()` to apply the result: a
`FusionWeights.coef` dict simply omits the held-out feature's key, and
`fuse()`'s own `coef.get(k, 0.0)` already treats a missing key as no
contribution - no new application logic needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from portcullis.core.fusion import FusionWeights

FEATURE_ORDER = ("l0", "l1", "l2", "knn")


def fit_fusion_variant(
    fit_features: np.ndarray, fit_labels: np.ndarray, *, include: tuple[str, ...]
) -> FusionWeights:
    indices = [FEATURE_ORDER.index(name) for name in include]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(fit_features[:, indices], fit_labels)
    coef = {name: float(clf.coef_[0][i]) for i, name in enumerate(include)}
    return FusionWeights(intercept=float(clf.intercept_[0]), coef=coef)
