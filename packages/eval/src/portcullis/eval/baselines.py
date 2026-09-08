"""The regex-only baseline (ADR-0010, Decision 1): L1's own score, used
alone with no L2/kNN/fusion at all. Not a separate scorer - L1's score is
already one column of a `ScoredPartition`'s feature matrix; this module
exists so "which column is the regex-only baseline" is named once, not
re-derived at each call site.
"""

from __future__ import annotations

import numpy as np

from .scoring import ScoredPartition

_L1_COLUMN = 1  # ScoredPartition.feature_matrix()'s fixed l0, l1, l2, knn order


def regex_only_scores(partition: ScoredPartition) -> np.ndarray:
    return partition.feature_matrix()[:, _L1_COLUMN]
