"""Full-cascade latency (ADR-0010): warm-up then N single-row timed
iterations through L0->L1->L2->kNN->fusion, unbatched - the same
methodology M4's `export.py` used for L2 alone (warm-up + N iterations,
p50/p95/p99), now over the whole cascade fusion actually combines.
Deliberately unbatched: a live request is one row, not a batch, and the
kNN sidecar's own `score()` (not `score_batch()`) is the method a real
per-request call path actually uses (gateway/pipeline.py's own pattern).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from portcullis.core.fusion import FusionWeights, LayerScores, fuse
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, Scope
from portcullis.training.knn.scorer import KnnScorer
from portcullis.training.l2.infer import L2Scorer

DEFAULT_WARMUP = 20


@dataclass(frozen=True, slots=True)
class LatencyPercentiles:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n: int


def _score_once(
    text: str,
    *,
    l1_engine: RuleEngine,
    l2_scorer: L2Scorer,
    knn_scorer: KnnScorer,
    fusion_weights: FusionWeights,
    scope: Scope,
) -> float:
    report = normalize(text)
    l1_result = l1_engine.evaluate(report, scope)
    l2_score = l2_scorer.score(text)
    knn_score = knn_scorer.score(text).score
    scores = LayerScores(
        l0=report.obfuscation_score, l1=l1_result.score, l2=l2_score, knn=knn_score
    )
    return fuse(scores, fusion_weights).score


def measure_cascade_latency(
    texts: list[str],
    *,
    l1_engine: RuleEngine,
    l2_scorer: L2Scorer,
    knn_scorer: KnnScorer,
    fusion_weights: FusionWeights,
    scope: Scope = Scope.USER,
    warmup: int = DEFAULT_WARMUP,
) -> LatencyPercentiles:
    for text in texts[:warmup]:
        _score_once(
            text,
            l1_engine=l1_engine,
            l2_scorer=l2_scorer,
            knn_scorer=knn_scorer,
            fusion_weights=fusion_weights,
            scope=scope,
        )

    latencies: list[float] = []
    for text in texts:
        started = time.perf_counter()
        _score_once(
            text,
            l1_engine=l1_engine,
            l2_scorer=l2_scorer,
            knn_scorer=knn_scorer,
            fusion_weights=fusion_weights,
            scope=scope,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)

    latencies.sort()
    n = len(latencies)
    return LatencyPercentiles(
        p50_ms=latencies[int(n * 0.50)],
        p95_ms=latencies[int(n * 0.95)],
        p99_ms=latencies[min(int(n * 0.99), n - 1)],
        n=n,
    )
