"""Computes one LayerScores per row by actually running L0, L1, L2 and the
kNN sidecar - the only place in the repo, as of Milestone 5, where all four
existing cascade layers are called in sequence. This is training-time
orchestration for fitting and evaluating fusion; a live gateway calling the
same four layers per-request is Milestone 6's job, not this module's.
"""

from __future__ import annotations

from portcullis.core.fusion import LayerScores
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, Scope

from ..knn.scorer import KnnScorer
from ..l2.infer import L2Scorer


def score_row(
    text: str,
    *,
    l1_engine: RuleEngine,
    l2_scorer: L2Scorer,
    knn_scorer: KnnScorer,
    scope: Scope = Scope.USER,
) -> LayerScores:
    report = normalize(text)
    l0 = report.obfuscation_score
    l1 = l1_engine.evaluate(report, scope).score
    l2 = l2_scorer.score(text)
    knn = knn_scorer.score(text).score
    return LayerScores(l0=l0, l1=l1, l2=l2, knn=knn)
