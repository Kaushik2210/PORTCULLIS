"""The live, per-request cascade: L0 -> L1 -> L2 -> kNN -> fusion -> policy.

training/fusion/features.py already does this same four-layer sequence for
offline fusion-fitting and said explicitly that a live gateway calling it
per-request was Milestone 6's job (docstring, written at Milestone 5). This
module is that job: same layers, same order, but instrumented per-layer for
the latency breakdown the standalone detection API promises, and returning
the full explainability payload (matched spans, taxonomy labels, nearest
known attack) rather than just the four raw scores fusion needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from portcullis.core.fusion import FusionWeights, LayerScores, fuse
from portcullis.core.l0 import normalize
from portcullis.core.l1 import RuleEngine, RuleMatch, Scope, load_rules_from_dir
from portcullis.core.policy import PolicyConfig, Verdict, decide
from portcullis.training.knn.embed import load_embedder
from portcullis.training.knn.scorer import KnnScorer
from portcullis.training.l2.infer import L2Scorer

from .config import GatewayConfig


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    l0_ms: float
    l1_ms: float
    l2_ms: float
    knn_ms: float
    fusion_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    text: str
    score: float
    verdict: Verdict
    enforced: bool
    rationale: str
    contributions: dict[str, float]
    matched_rules: tuple[RuleMatch, ...]
    taxonomy_labels: tuple[str, ...]
    obfuscation_score: float
    max_decode_depth: int
    nearest_attack_text: str | None
    nearest_attack_family: str | None
    latency: LatencyBreakdown


class DetectionPipeline:
    """Loads every layer once; scores many requests cheaply after that.

    Construction pays the real cost (ONNX session, sentence-transformer,
    kNN index load) exactly once at process startup - the same reason
    L2Scorer and KnnScorer themselves load-once-score-many (see their own
    docstrings). A gateway that reloaded these per request would blow the
    latency budget on model loading alone.
    """

    def __init__(
        self,
        *,
        l1_engine: RuleEngine,
        l2_scorer: L2Scorer,
        knn_scorer: KnnScorer,
        fusion_weights: FusionWeights,
        policy_config: PolicyConfig,
    ) -> None:
        self._l1_engine = l1_engine
        self._l2_scorer = l2_scorer
        self._knn_scorer = knn_scorer
        self._fusion_weights = fusion_weights
        self._policy_config = policy_config

    @classmethod
    def from_config(cls, config: GatewayConfig) -> DetectionPipeline:
        l1_engine = RuleEngine(load_rules_from_dir(config.rules_dir))
        l2_scorer = L2Scorer.from_checkpoint(config.checkpoint_dir, config.onnx_path)
        embedder = load_embedder()
        knn_scorer = KnnScorer.from_saved_index(config.knn_index_dir, embedder)
        fusion_weights = FusionWeights.load(config.fusion_weights_path)
        policy_config = PolicyConfig(
            flag_threshold=config.flag_threshold,
            sanitise_threshold=config.sanitise_threshold,
            challenge_threshold=config.challenge_threshold,
            block_threshold=config.block_threshold,
        )
        return cls(
            l1_engine=l1_engine,
            l2_scorer=l2_scorer,
            knn_scorer=knn_scorer,
            fusion_weights=fusion_weights,
            policy_config=policy_config,
        )

    def detect(
        self, text: str, *, scope: Scope = Scope.USER, shadow: bool = False
    ) -> DetectionResult:
        started = time.perf_counter()

        t = time.perf_counter()
        report = normalize(text)
        l0_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        l1_result = self._l1_engine.evaluate(report, scope)
        l1_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        l2_score = self._l2_scorer.score(text)
        l2_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        knn_result = self._knn_scorer.score(text)
        knn_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        scores = LayerScores(
            l0=report.obfuscation_score, l1=l1_result.score, l2=l2_score, knn=knn_result.score
        )
        fusion_result = fuse(scores, self._fusion_weights)
        policy_result = decide(fusion_result.score, self._policy_config, shadow=shadow)
        fusion_ms = (time.perf_counter() - t) * 1000

        total_ms = (time.perf_counter() - started) * 1000

        taxonomy_labels = tuple(sorted({label for m in l1_result.matches for label in m.labels}))

        return DetectionResult(
            text=text,
            score=fusion_result.score,
            verdict=policy_result.verdict,
            enforced=policy_result.enforced,
            rationale=policy_result.rationale,
            contributions=fusion_result.contributions,
            matched_rules=l1_result.matches,
            taxonomy_labels=taxonomy_labels,
            obfuscation_score=report.obfuscation_score,
            max_decode_depth=report.max_decode_depth,
            nearest_attack_text=knn_result.nearest.entry.text if knn_result.nearest else None,
            nearest_attack_family=(knn_result.nearest.entry.family if knn_result.nearest else None),
            latency=LatencyBreakdown(
                l0_ms=l0_ms,
                l1_ms=l1_ms,
                l2_ms=l2_ms,
                knn_ms=knn_ms,
                fusion_ms=fusion_ms,
                total_ms=total_ms,
            ),
        )
