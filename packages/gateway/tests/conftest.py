"""A fake DetectionPipeline for tests that exercise the gateway's routing,
proxying and blocking behaviour without loading the real ONNX/
sentence-transformer checkpoint. Real-pipeline tests are `needs_model` and
live in test_pipeline_real.py.

`FakeDetectionPipeline` is a genuine `DetectionPipeline` subclass (not just
a duck-typed look-alike) so `create_app(pipeline=...)`'s type stays
`DetectionPipeline | None` in both production and tests - the fake's
`__init__` deliberately skips calling `super().__init__()`, since that
would require the four real layer objects this fixture exists to avoid
constructing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from portcullis.core.l0 import ViewKind
from portcullis.core.l1 import RuleMatch, Scope, Severity
from portcullis.core.l4 import InMemoryConversationStore
from portcullis.core.policy import Verdict
from portcullis.gateway.conversation import ConversationAwareDetector
from portcullis.gateway.mock_upstream import app as mock_upstream_app
from portcullis.gateway.pipeline import DetectionPipeline, DetectionResult, LatencyBreakdown

# Deterministic stand-in for L2/kNN: any text containing this substring is
# "detected" as an attack, so tests can exercise both the ALLOW and BLOCK
# paths without a real classifier.
INJECTION_MARKER = "IGNORE ALL PREVIOUS INSTRUCTIONS"


class FakeDetectionPipeline(DetectionPipeline):
    def __init__(self) -> None:  # real layers deliberately not constructed
        pass

    def detect(
        self, text: str, *, scope: Scope = Scope.USER, shadow: bool = False
    ) -> DetectionResult:
        is_attack = INJECTION_MARKER in text
        score = 0.95 if is_attack else 0.05
        verdict = Verdict.BLOCK if is_attack else Verdict.ALLOW
        matched_rules = (
            (
                RuleMatch(
                    rule_id="fake-001",
                    name="fake override phrase",
                    severity=Severity.CRITICAL,
                    labels=("direct_override",),
                    weight=0.95,
                    view=ViewKind.CANONICAL,
                    span=(0, len(text)),
                    matched_text=text[:200],
                    rationale="fake-001 (fake override phrase): matched for test purposes",
                ),
            )
            if is_attack
            else ()
        )
        return DetectionResult(
            text=text,
            score=score,
            verdict=verdict,
            enforced=not shadow,
            rationale=f"fake pipeline: score {score:.2f}",
            contributions={"l0": 0.0, "l1": score, "l2": score, "knn": 0.0},
            matched_rules=matched_rules,
            taxonomy_labels=("direct_override",) if is_attack else (),
            obfuscation_score=0.0,
            max_decode_depth=0,
            nearest_attack_text=INJECTION_MARKER if is_attack else None,
            nearest_attack_family="direct_override" if is_attack else None,
            latency=LatencyBreakdown(
                l0_ms=0.1, l1_ms=0.1, l2_ms=0.1, knn_ms=0.1, fusion_ms=0.1, total_ms=0.5
            ),
        )


@pytest.fixture
def fake_pipeline() -> FakeDetectionPipeline:
    return FakeDetectionPipeline()


@pytest.fixture
def injection_marker() -> str:
    return INJECTION_MARKER


def fake_embed_fn(text: str) -> tuple[float, ...]:
    """Not semantically meaningful - deterministic and exactly controllable
    so a wiring test can craft "same topic" vs. "different topic" pairs by
    construction, rather than relying on a real embedder's actual semantic
    behaviour (which core's own L4 test suite already covers exhaustively
    with hand-picked similarity values)."""
    return (1.0, 0.0) if "TOPIC_A" in text else (0.0, 1.0)


@pytest.fixture
def conversation_detector(fake_pipeline: FakeDetectionPipeline) -> ConversationAwareDetector:
    return ConversationAwareDetector(
        fake_pipeline, InMemoryConversationStore(), fake_embed_fn, ttl_s=1800.0
    )


@pytest.fixture
def mock_upstream_asgi_app() -> FastAPI:
    return mock_upstream_app
