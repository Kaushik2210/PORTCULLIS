"""Milestone 9's checkpoint artifact: the real benchmark table. Scores the
real ~20k-row test set through the actually-deployed cascade (loading M5's
own fitted `fusion_weights.json`, not a fresh refit - the headline number
describes the system that runs, not a variant of it), computes the
headline metrics with bootstrap CIs, the regex-only and max() baselines,
the L0/kNN ablations, full-cascade latency, and two adaptive-attacker
techniques. ADR-0010 has the full methodology and the scope this milestone
deliberately does not cover.

Run with `just eval-m9` (writes docs/benchmarks/eval-results.json).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from portcullis.core.fusion import FusionWeights, LayerScores, fuse, max_baseline
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, Scope, load_rules_from_dir
from portcullis.training.knn.embed import load_embedder
from portcullis.training.knn.scorer import KnnScorer
from portcullis.training.l2.infer import L2Scorer

from .adaptive import generate_paraphrases, run_blackbox_attack
from .baselines import regex_only_scores
from .fusion_variants import fit_fusion_variant
from .latency import DEFAULT_WARMUP, measure_cascade_latency
from .metrics import auprc, bootstrap_tpr_at_fpr
from .scoring import ScoredRow, score_partition

_LOCAL_APPDATA = Path.home() / "AppData" / "Local" / "portcullis"
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_RULES_DIR = Path("rules")
DEFAULT_CHECKPOINT_DIR = _LOCAL_APPDATA / "l2_checkpoint"
DEFAULT_ONNX_PATH = _LOCAL_APPDATA / "l2_export" / "l2.int8.onnx"
DEFAULT_KNN_INDEX_DIR = _LOCAL_APPDATA / "knn_index"

TARGET_FPRS = (0.001, 0.01)
BLOCK_THRESHOLD = (
    0.8  # matches the gateway's own default (config.py) - real enforcement, not an abstract point
)
LATENCY_SAMPLE_SIZE = 500
ADAPTIVE_PARAPHRASE_SEEDS = 200
ADAPTIVE_BLACKBOX_SEEDS = 100
ADAPTIVE_BLACKBOX_MAX_QUERIES = 100


def _fpr_key(fpr: float) -> str:
    return f"{fpr:.3%}".rstrip("0").rstrip(".") if fpr < 0.01 else f"{fpr:.1%}"


def _bootstrap_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {"auprc": auprc(y_true, y_score)}
    for target_fpr in TARGET_FPRS:
        r = bootstrap_tpr_at_fpr(y_true, y_score, target_fpr)
        result[f"tpr_at_fpr_{_fpr_key(target_fpr)}"] = {
            "tpr": r.tpr,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "ci_level": r.ci_level,
            "n_resamples": r.n_resamples,
        }
    return result


def run(
    *,
    data_dir: Path,
    rules_dir: Path,
    checkpoint_dir: Path,
    onnx_path: Path,
    knn_index_dir: Path,
    out_dir: Path,
) -> None:
    started = time.perf_counter()

    print("loading L1 rule engine, L2 scorer, kNN sidecar", flush=True)
    l1_engine = RuleEngine(load_rules_from_dir(rules_dir))
    l2_scorer = L2Scorer.from_checkpoint(checkpoint_dir, onnx_path)
    embedder = load_embedder()
    knn_scorer = KnnScorer.from_saved_index(knn_index_dir, embedder)
    fusion_weights = FusionWeights.load(checkpoint_dir / "fusion_weights.json")
    print(f"  loaded deployed fusion weights: {fusion_weights.coef}", flush=True)

    print("scoring validation (ablation fit set)", flush=True)
    val_scored = score_partition(
        data_dir / "validation.jsonl",
        l1_engine=l1_engine,
        l2_scorer=l2_scorer,
        knn_scorer=knn_scorer,
        log_label="validation",
    )
    print("scoring test (headline eval set - the real ~20k rows, not a sample)", flush=True)
    test_scored = score_partition(
        data_dir / "test.jsonl",
        l1_engine=l1_engine,
        l2_scorer=l2_scorer,
        knn_scorer=knn_scorer,
        log_label="test",
    )

    y_test = test_scored.labels

    print("computing headline metrics (deployed fusion weights)", flush=True)
    fused_scores = np.array([fuse(row.scores, fusion_weights).score for row in test_scored.rows])
    headline = _bootstrap_report(y_test, fused_scores)

    print("computing max() baseline", flush=True)
    max_scores = np.array([max_baseline(row.scores).value for row in test_scored.rows])
    max_report = _bootstrap_report(y_test, max_scores)

    print("computing regex-only baseline", flush=True)
    regex_scores = regex_only_scores(test_scored)
    regex_report = _bootstrap_report(y_test, regex_scores)

    print("fitting and evaluating ablations", flush=True)
    val_features = val_scored.feature_matrix()
    val_labels = val_scored.labels
    ablations: dict[str, object] = {}
    ablation_scores: dict[str, np.ndarray] = {}
    for name, include in (
        ("remove_l0", ("l1", "l2", "knn")),
        ("remove_knn", ("l0", "l1", "l2")),
    ):
        weights = fit_fusion_variant(val_features, val_labels, include=include)
        scores = np.array([fuse(row.scores, weights).score for row in test_scored.rows])
        ablations[name] = {"coef": weights.coef, **_bootstrap_report(y_test, scores)}
        ablation_scores[name] = scores

    print(f"measuring full-cascade latency (n={LATENCY_SAMPLE_SIZE}, unbatched)", flush=True)
    rng = random.Random(0)  # noqa: S311 - sampling for a latency measurement, not cryptography
    latency_texts = [row.text for row in rng.sample(test_scored.rows, LATENCY_SAMPLE_SIZE)]
    latency = measure_cascade_latency(
        latency_texts,
        l1_engine=l1_engine,
        l2_scorer=l2_scorer,
        knn_scorer=knn_scorer,
        fusion_weights=fusion_weights,
        warmup=DEFAULT_WARMUP,
    )

    def score_text(text: str) -> float:
        row_scores = _score_row(
            text, l1_engine=l1_engine, l2_scorer=l2_scorer, knn_scorer=knn_scorer
        )
        return fuse(row_scores, fusion_weights).score

    attack_rows = [row for row in test_scored.rows if row.label == 1]

    print(f"running paraphrase attacks (n={ADAPTIVE_PARAPHRASE_SEEDS} seeds)", flush=True)
    paraphrase_seeds = rng.sample(attack_rows, min(ADAPTIVE_PARAPHRASE_SEEDS, len(attack_rows)))
    paraphrase_results = _run_paraphrase_attacks(paraphrase_seeds, score_text)

    print(f"running black-box query attacks (n={ADAPTIVE_BLACKBOX_SEEDS} seeds)", flush=True)
    blackbox_seeds = rng.sample(attack_rows, min(ADAPTIVE_BLACKBOX_SEEDS, len(attack_rows)))
    blackbox_results = [
        run_blackbox_attack(
            row.text,
            score_text,
            block_threshold=BLOCK_THRESHOLD,
            max_queries=ADAPTIVE_BLACKBOX_MAX_QUERIES,
            rng_seed=i,
        )
        for i, row in enumerate(blackbox_seeds)
    ]
    blackbox_evasion_rate = sum(r.evaded for r in blackbox_results) / len(blackbox_results)

    report = {
        "test_set": {"n": len(test_scored.rows), "n_attack": len(attack_rows)},
        "fusion_weights": {"intercept": fusion_weights.intercept, "coef": fusion_weights.coef},
        "headline": headline,
        "baselines": {"max": max_report, "regex_only": regex_report},
        "ablations": ablations,
        "latency_ms": {
            "p50": latency.p50_ms,
            "p95": latency.p95_ms,
            "p99": latency.p99_ms,
            "n": latency.n,
        },
        "adaptive": {
            "paraphrase": paraphrase_results,
            "blackbox_query": {
                "n_seeds": len(blackbox_results),
                "evasion_rate": blackbox_evasion_rate,
                "block_threshold": BLOCK_THRESHOLD,
                "max_queries": ADAPTIVE_BLACKBOX_MAX_QUERIES,
                "mean_queries_used": float(np.mean([r.queries_used for r in blackbox_results])),
            },
        },
        "elapsed_s": time.perf_counter() - started,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval-results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)

    # Milestone 10's dashboard needs per-row (label, score) pairs, not just
    # the aggregate metrics above - the threshold slider recomputes a
    # confusion matrix client-side from these, and the red-team console's
    # "diff two model versions" reuses remove_knn as the second version
    # (ADR-0011, Decisions 2 and 3) rather than a second trained checkpoint.
    scores_out = {
        "labels": y_test.tolist(),
        "fusion": fused_scores.tolist(),
        "remove_knn": ablation_scores["remove_knn"].tolist(),
    }
    scores_path = out_dir / "eval-scores.json"
    scores_path.write_text(json.dumps(scores_out), encoding="utf-8")
    print(f"wrote {scores_path}", flush=True)
    print(f"total elapsed: {report['elapsed_s']:.1f}s", flush=True)


def _score_row(
    text: str, *, l1_engine: RuleEngine, l2_scorer: L2Scorer, knn_scorer: KnnScorer
) -> LayerScores:
    report = normalize(text)
    l1_result = l1_engine.evaluate(report, Scope.USER)
    l2_score = l2_scorer.score(text)
    knn_score = knn_scorer.score(text).score
    return LayerScores(l0=report.obfuscation_score, l1=l1_result.score, l2=l2_score, knn=knn_score)


def _run_paraphrase_attacks(
    seed_rows: list[ScoredRow], score_text: Callable[[str], float]
) -> dict[str, object]:
    total_variants = 0
    evaded_variants = 0
    seeds_with_any_evasion = 0
    seeds_with_no_paraphrase_available = 0

    for row in seed_rows:
        variants = generate_paraphrases(row.text)
        if not variants:
            seeds_with_no_paraphrase_available += 1
            continue
        seed_evaded = False
        for variant in variants:
            total_variants += 1
            if score_text(variant.paraphrased) < BLOCK_THRESHOLD:
                evaded_variants += 1
                seed_evaded = True
        if seed_evaded:
            seeds_with_any_evasion += 1

    return {
        "n_seeds": len(seed_rows),
        "n_seeds_with_no_paraphrase_available": seeds_with_no_paraphrase_available,
        "n_variants_tried": total_variants,
        "variant_evasion_rate": evaded_variants / total_variants if total_variants else 0.0,
        "seed_evasion_rate": seeds_with_any_evasion / len(seed_rows) if seed_rows else 0.0,
        "block_threshold": BLOCK_THRESHOLD,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    ap.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    ap.add_argument("--knn-index-dir", type=Path, default=DEFAULT_KNN_INDEX_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/benchmarks"))
    args = ap.parse_args()
    run(
        data_dir=args.data_dir,
        rules_dir=args.rules_dir,
        checkpoint_dir=args.checkpoint_dir,
        onnx_path=args.onnx_path,
        knn_index_dir=args.knn_index_dir,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
