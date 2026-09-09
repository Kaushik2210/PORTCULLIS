"""Serves Milestone 9's real eval artifacts to the dashboard (ADR-0011,
Decisions 2 and 3): `eval-scores.json`'s per-row (label, score) pairs for
the threshold slider's client-side confusion matrix, and
`eval-results.json`'s aggregate metrics (including the `remove_knn`
ablation the red-team console diffs against as its "second model
version"). Loaded once, held in memory, re-served on every request - these
files don't change while the gateway process is running.

Missing files degrade to `None` here (and a 404 at the route), not a
startup crash - a fresh clone, or a gateway started before `just eval` has
ever run, should still come up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


class EvalDataStore:
    def __init__(self, *, results_path: Path, scores_path: Path) -> None:
        self._results = _load_json(results_path)
        self._scores = _load_json(scores_path)

    @property
    def results(self) -> dict[str, Any] | None:
        return self._results

    @property
    def scores(self) -> dict[str, Any] | None:
        return self._scores
