"""Fits fusion weights and produces the fusion-vs-max() comparison - the
Milestone 5 checkpoint artifact the spec asks for.

Split discipline, matching the rest of this project (M3's leakage audit, M4's
calibration on validation rather than train): fusion is **fit** on
`validation` and **evaluated** on a held-out slice of `test`, never the same
rows. Fitting on train would be circular - L2's classifier weights were
themselves fit on train (Milestone 4's 0.052 final loss means it has
memorised much of it), so train rows are not a fair test of what fusion
learns to trust about L2's score. Validation was only used for L2's
*calibration* (a much lower-capacity fit: two numbers), so it is still a
fair, uncontaminated set for fitting fusion's four coefficients.

`test` is large (19,937 rows); scoring it in full - four layers per row,
including an ONNX forward pass and a sentence-embedding call each - would
cost real wall-clock for a Milestone-5 proof rather than Milestone 9's
rigorous, bootstrap-CI'd harness. A deterministic stride sample (same
technique as train.py's --max-train-rows, for the same reason: test.jsonl
is sorted by row id, PromptShield's original heavily-templated order, and a
prefix would risk over/under-representing whatever templates sit first)
keeps this milestone's runtime bounded while still being a real, honest
measurement - not the headline number, a checkpoint on the way to it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from portcullis.core.fusion import FusionWeights, LayerScores, fuse, max_baseline
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, Scope, load_rules_from_dir

from ..knn.embed import load_embedder
from ..knn.scorer import KnnScorer
from ..l2.infer import L2Scorer

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_RULES_DIR = Path("rules")
DEFAULT_TEST_SAMPLE = 3000
TARGET_FPR = 0.01  # 1% - a Milestone-5 proof point, not the M9 0.1% headline


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stride_sample(rows: list[dict[str, object]], n: int) -> list[dict[str, object]]:
    if n >= len(rows):
        return rows
    stride = len(rows) / n
    return [rows[int(i * stride)] for i in range(n)]


def _score_all(
    rows: list[dict[str, object]],
    *,
    l1_engine: RuleEngine,
    l2_scorer: L2Scorer,
    knn_scorer: KnnScorer,
    log_every: int = 500,
) -> tuple[list[LayerScores], list[int]]:
    """L0/L1/L2 stay per-row (all three were already fast enough per-call -
    L2's ONNX INT8 forward pass measured ~12.6ms/row at Milestone 4). The
    kNN sidecar's embedding step is batched up front instead: embedding one
    text at a time through sentence-transformers pays a fixed per-call
    overhead that batching amortises away, and running this loop the naive
    per-row way was measured at 2.1 rows/s before this fix - a ~30 minute
    run for what batching brings down to a few minutes.
    """
    texts = [str(r["text"]) for r in rows]
    print(f"  batch-embedding {len(texts)} rows for the kNN sidecar", flush=True)
    knn_scores = knn_scorer.score_batch(texts)

    scores: list[LayerScores] = []
    labels: list[int] = []
    started = time.perf_counter()
    for i, (row, knn) in enumerate(zip(rows, knn_scores, strict=True), start=1):
        report = normalize(str(row["text"]))
        l0 = report.obfuscation_score
        l1 = l1_engine.evaluate(report, Scope.USER).score
        l2 = l2_scorer.score(str(row["text"]))
        scores.append(LayerScores(l0=l0, l1=l1, l2=l2, knn=knn.score))
        labels.append(int(row["label"]))  # type: ignore[call-overload]
        if i % log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  scored {i}/{len(rows)} ({i / elapsed:.1f} rows/s)", flush=True)
    return scores, labels


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the operating point where FPR is closest to (and not above)
    target_fpr - the same fixed-FPR framing NFR-3 requires for the real
    headline metric, applied here at a coarser 1% point this small a sample
    can actually support (see module docstring: this is not M9's number)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = fpr <= target_fpr
    if not eligible.any():
        return 0.0
    return float(tpr[eligible].max())


def run(
    *,
    data_dir: Path,
    rules_dir: Path,
    knn_index_dir: Path,
    checkpoint_dir: Path,
    onnx_path: Path,
    out_dir: Path,
    test_sample: int,
) -> None:
    print("loading L1 rule engine", flush=True)
    l1_engine = RuleEngine(load_rules_from_dir(rules_dir))

    print("loading L2 scorer (ONNX + Platt calibration)", flush=True)
    l2_scorer = L2Scorer.from_checkpoint(checkpoint_dir, onnx_path)

    print("loading kNN sidecar (saved index + embedder)", flush=True)
    embedder = load_embedder()
    knn_scorer = KnnScorer.from_saved_index(knn_index_dir, embedder)

    print("scoring validation (fusion fit set)", flush=True)
    val_rows = _load_jsonl(data_dir / "validation.jsonl")
    val_scores, val_labels = _score_all(
        val_rows, l1_engine=l1_engine, l2_scorer=l2_scorer, knn_scorer=knn_scorer
    )

    print(f"scoring test sample (fusion eval set, n={test_sample})", flush=True)
    test_rows = _stride_sample(_load_jsonl(data_dir / "test.jsonl"), test_sample)
    test_scores, test_labels = _score_all(
        test_rows, l1_engine=l1_engine, l2_scorer=l2_scorer, knn_scorer=knn_scorer
    )

    print("fitting fusion (logistic regression on validation)", flush=True)
    fit_features = np.array([[s.l0, s.l1, s.l2, s.knn] for s in val_scores])
    fit_labels = np.array(val_labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(fit_features, fit_labels)
    weights = FusionWeights(
        intercept=float(clf.intercept_[0]),
        coef={
            "l0": float(clf.coef_[0][0]),
            "l1": float(clf.coef_[0][1]),
            "l2": float(clf.coef_[0][2]),
            "knn": float(clf.coef_[0][3]),
        },
    )
    print(f"  weights: intercept={weights.intercept:.4f} coef={weights.coef}", flush=True)

    print("evaluating fusion vs max() on held-out test sample", flush=True)
    y_test = np.array(test_labels)
    fused_scores = np.array([fuse(s, weights).score for s in test_scores])
    max_scores = np.array([max_baseline(s).value for s in test_scores])

    fused_tpr = _tpr_at_fpr(y_test, fused_scores, TARGET_FPR)
    max_tpr = _tpr_at_fpr(y_test, max_scores, TARGET_FPR)

    from sklearn.metrics import average_precision_score

    fused_auprc = float(average_precision_score(y_test, fused_scores))
    max_auprc = float(average_precision_score(y_test, max_scores))

    print(
        f"  TPR@{TARGET_FPR:.0%}FPR: fused={fused_tpr:.4f} max={max_tpr:.4f} "
        f"delta={fused_tpr - max_tpr:+.4f}",
        flush=True,
    )
    print(f"  AUPRC: fused={fused_auprc:.4f} max={max_auprc:.4f}", flush=True)

    weights.save(out_dir / "fusion_weights.json")
    report = {
        "fit_set": {"partition": "validation", "n": len(val_rows)},
        "eval_set": {"partition": "test", "n": len(test_rows), "sampling": "deterministic stride"},
        "target_fpr": TARGET_FPR,
        "fused": {"tpr_at_target_fpr": fused_tpr, "auprc": fused_auprc},
        "max_baseline": {"tpr_at_target_fpr": max_tpr, "auprc": max_auprc},
        "delta": {"tpr": fused_tpr - max_tpr, "auprc": fused_auprc - max_auprc},
        "weights": {"intercept": weights.intercept, "coef": weights.coef},
    }
    report_path = Path("docs/benchmarks/fusion-comparison.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'fusion_weights.json'}", flush=True)
    print(f"wrote {report_path}", flush=True)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    ap.add_argument("--knn-index-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--onnx-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--test-sample", type=int, default=DEFAULT_TEST_SAMPLE)
    args = ap.parse_args()
    run(
        data_dir=args.data_dir,
        rules_dir=args.rules_dir,
        knn_index_dir=args.knn_index_dir,
        checkpoint_dir=args.checkpoint_dir,
        onnx_path=args.onnx_path,
        out_dir=args.out_dir,
        test_sample=args.test_sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
