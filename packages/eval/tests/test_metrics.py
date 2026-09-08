"""tpr_at_fpr, bootstrap_tpr_at_fpr and auprc against synthetic arrays with
known properties - the statistics have to be right before anything is
trusted to run over the real ~20k-row test set (ADR-0010).
"""

from __future__ import annotations

import numpy as np
import pytest

from portcullis.eval.metrics import auprc, bootstrap_tpr_at_fpr, tpr_at_fpr


def _perfect_separation(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0] * n + [1] * n)
    y_score = np.array([0.1] * n + [0.9] * n)
    return y_true, y_score


def _no_separation(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(n)
    return y_true, y_score


def test_perfect_separation_scores_full_tpr_at_any_reasonable_fpr() -> None:
    y_true, y_score = _perfect_separation()
    assert tpr_at_fpr(y_true, y_score, target_fpr=0.01) == pytest.approx(1.0)


def test_no_separation_gives_a_low_tpr_at_a_low_fpr() -> None:
    y_true, y_score = _no_separation()
    result = tpr_at_fpr(y_true, y_score, target_fpr=0.01)
    # Random scores: TPR at a fixed low FPR should itself be low - it must
    # not be anywhere near 1.0, which would indicate a bug, not a good
    # detector.
    assert result < 0.15


def test_impossible_fpr_target_returns_zero_not_an_error() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])  # no threshold separates anything
    result = tpr_at_fpr(y_true, y_score, target_fpr=0.0)
    assert result == 0.0


def test_bootstrap_ci_contains_the_point_estimate() -> None:
    y_true, y_score = _perfect_separation()
    result = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr=0.05, n_resamples=200)
    assert result.ci_low <= result.tpr <= result.ci_high


def test_bootstrap_ci_is_narrower_with_more_data() -> None:
    small_true, small_score = _no_separation(n=100, seed=1)
    large_true, large_score = _no_separation(n=5000, seed=1)
    small = bootstrap_tpr_at_fpr(small_true, small_score, target_fpr=0.1, n_resamples=500, seed=42)
    large = bootstrap_tpr_at_fpr(large_true, large_score, target_fpr=0.1, n_resamples=500, seed=42)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_bootstrap_is_reproducible_with_the_same_seed() -> None:
    y_true, y_score = _no_separation()
    a = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr=0.05, n_resamples=300, seed=7)
    b = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr=0.05, n_resamples=300, seed=7)
    assert a == b


def test_different_seeds_can_give_different_resamples() -> None:
    y_true, y_score = _no_separation()
    a = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr=0.05, n_resamples=300, seed=1)
    b = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr=0.05, n_resamples=300, seed=2)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_auprc_is_high_for_perfect_separation() -> None:
    y_true, y_score = _perfect_separation()
    assert auprc(y_true, y_score) == pytest.approx(1.0)


def test_auprc_for_random_scores_is_near_class_prevalence() -> None:
    y_true, y_score = _no_separation(n=5000)
    prevalence = y_true.mean()
    # AUPRC for a non-informative scorer converges to the positive
    # prevalence - a generous tolerance since this is a finite sample.
    assert abs(auprc(y_true, y_score) - prevalence) < 0.05
