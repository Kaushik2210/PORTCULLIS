"""Scores every row of a JSONL partition through the real L0->L1->L2->kNN
cascade - the one place this milestone's numbers come from real inference,
not a projection. Batches the kNN sidecar's embedding calls up front, the
same fix M5's fusion-fitting needed for the same reason (embedding one
text at a time measured 2.1 rows/s; batched, >100 rows/s) - scoring the
full ~20k-row test set once is this milestone's single most expensive
step, and it only needs to happen once (metrics.py's bootstrap resamples
the resulting array, not the model).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from portcullis.core.fusion import LayerScores
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, Scope
from portcullis.training.knn.scorer import KnnScorer
from portcullis.training.l2.infer import L2Scorer


@dataclass(frozen=True, slots=True)
class ScoredRow:
    id: str
    label: int
    text: str
    scores: LayerScores


@dataclass(frozen=True, slots=True)
class ScoredPartition:
    rows: tuple[ScoredRow, ...]

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.rows])

    def feature_matrix(self) -> np.ndarray:
        """Column order l0, l1, l2, knn - matches `LayerScores`'s own
        field order and `FusionWeights.coef` keys everywhere else in the
        project (fusion/fit.py, gateway/pipeline.py)."""
        return np.array([[r.scores.l0, r.scores.l1, r.scores.l2, r.scores.knn] for r in self.rows])


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_partition(
    path: Path,
    *,
    l1_engine: RuleEngine,
    l2_scorer: L2Scorer,
    knn_scorer: KnnScorer,
    scope: Scope = Scope.USER,
    log_every: int = 2000,
    log_label: str = "",
) -> ScoredPartition:
    raw_rows = _load_rows(path)
    texts = [str(r["text"]) for r in raw_rows]

    print(f"[{log_label or path.name}] batch-embedding {len(texts)} rows for kNN", flush=True)
    knn_scores = knn_scorer.score_batch(texts)

    scored: list[ScoredRow] = []
    started = time.perf_counter()
    for i, (row, knn) in enumerate(zip(raw_rows, knn_scores, strict=True), start=1):
        text = str(row["text"])
        report = normalize(text)
        l1_result = l1_engine.evaluate(report, scope)
        l2_score = l2_scorer.score(text)
        scored.append(
            ScoredRow(
                id=str(row["id"]),
                label=int(row["label"]),  # type: ignore[call-overload]
                text=text,
                scores=LayerScores(
                    l0=report.obfuscation_score, l1=l1_result.score, l2=l2_score, knn=knn.score
                ),
            )
        )
        if i % log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[{log_label or path.name}] scored {i}/{len(raw_rows)} ({i / elapsed:.1f} rows/s)",
                flush=True,
            )

    return ScoredPartition(rows=tuple(scored))
