"""Evaluation statistics: TPR at a fixed FPR (with a bootstrap CI) and
AUPRC. Pure numpy/sklearn operations over an already-scored `(label,
score)` array - scoring the corpus through the real cascade happens once
(scoring.py); everything here is cheap resampling on top of that, which is
what makes a 2000-resample bootstrap tractable (ADR-0010, Decision 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

DEFAULT_N_RESAMPLES = 2000
DEFAULT_CI = 0.95


@dataclass(frozen=True, slots=True)
class TprAtFprResult:
    tpr: float
    target_fpr: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the operating point where FPR is closest to (and not above)
    `target_fpr`. Returns 0.0 if no threshold achieves an FPR that low -
    an honest answer, not an error, for a target below what the ROC curve
    can resolve at this sample size."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = fpr <= target_fpr
    if not eligible.any():
        return 0.0
    return float(tpr[eligible].max())


def bootstrap_tpr_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> TprAtFprResult:
    """The point estimate plus a bootstrap CI: resample `(y_true, y_score)`
    pairs with replacement `n_resamples` times, recompute `tpr_at_fpr` on
    each resample, report the `ci` percentile interval. Seeded for
    reproducibility - the same seed on the same scored array reproduces
    the same CI, matching the project's reproducibility bar (NFR-6)."""
    point_estimate = tpr_at_fpr(y_true, y_score, target_fpr)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    resampled_tprs = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resampled_tprs[i] = tpr_at_fpr(y_true[indices], y_score[indices], target_fpr)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(resampled_tprs, alpha))
    ci_high = float(np.quantile(resampled_tprs, 1.0 - alpha))

    return TprAtFprResult(
        tpr=point_estimate,
        target_fpr=target_fpr,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        n_resamples=n_resamples,
    )


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))
