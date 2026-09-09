"""Gateway configuration: where the M4/M5 artifacts live, the policy
thresholds to enforce, and what upstream to proxy allowed traffic to.

Read from environment variables with defaults that match the rest of the
project's convention (heavy artifacts under %LOCALAPPDATA%\\portcullis,
never inside the OneDrive-synced repo - see the M0 gitignore/artifact-path
notes). Nothing here is a secret; there is no credential to load, since the
default upstream is the mock (ADR-0007).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_LOCAL_APPDATA = Path.home() / "AppData" / "Local" / "portcullis"


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    rules_dir: Path
    checkpoint_dir: Path
    onnx_path: Path
    knn_index_dir: Path
    fusion_weights_path: Path
    upstream_base_url: str
    shadow_mode: bool
    flag_threshold: float
    sanitise_threshold: float
    challenge_threshold: float
    block_threshold: float
    redis_url: str | None
    conversation_ttl_s: float
    eval_results_path: Path
    eval_scores_path: Path
    dashboard_origin: str

    @classmethod
    def from_env(cls) -> GatewayConfig:
        checkpoint_dir = Path(
            os.environ.get("PORTCULLIS_L2_CHECKPOINT_DIR", str(_LOCAL_APPDATA / "l2_checkpoint"))
        )
        return cls(
            rules_dir=Path(os.environ.get("PORTCULLIS_RULES_DIR", "rules")),
            checkpoint_dir=checkpoint_dir,
            onnx_path=Path(
                os.environ.get(
                    "PORTCULLIS_L2_ONNX_PATH",
                    str(_LOCAL_APPDATA / "l2_export" / "l2.int8.onnx"),
                )
            ),
            knn_index_dir=Path(
                os.environ.get("PORTCULLIS_KNN_INDEX_DIR", str(_LOCAL_APPDATA / "knn_index"))
            ),
            fusion_weights_path=Path(
                os.environ.get(
                    "PORTCULLIS_FUSION_WEIGHTS_PATH",
                    str(checkpoint_dir / "fusion_weights.json"),
                )
            ),
            upstream_base_url=os.environ.get(
                "PORTCULLIS_UPSTREAM_BASE_URL", "http://127.0.0.1:8000"
            ),
            shadow_mode=os.environ.get("PORTCULLIS_SHADOW_MODE", "false").lower() == "true",
            # Defaults chosen to land near the M5 fusion score's own class
            # balance (docs/benchmarks/fusion-comparison.json) rather than
            # picked arbitrarily - they are a starting point for a demo, not
            # a tuned operating point. A real deployment sets these from the
            # eval harness (Milestone 9), per tenant/route (ADR-0006).
            flag_threshold=float(os.environ.get("PORTCULLIS_FLAG_THRESHOLD", "0.2")),
            sanitise_threshold=float(os.environ.get("PORTCULLIS_SANITISE_THRESHOLD", "0.4")),
            challenge_threshold=float(os.environ.get("PORTCULLIS_CHALLENGE_THRESHOLD", "0.6")),
            block_threshold=float(os.environ.get("PORTCULLIS_BLOCK_THRESHOLD", "0.8")),
            # None (the default) means conversation tracking uses an
            # in-process InMemoryConversationStore - real, working L4, just
            # not shared across processes or surviving a restart. Set this
            # to opt into the Redis-backed store (ADR-0008, Decision 4).
            redis_url=os.environ.get("PORTCULLIS_REDIS_URL"),
            conversation_ttl_s=float(os.environ.get("PORTCULLIS_CONVERSATION_TTL_S", "1800")),
            # Milestone 9's real artifacts (ADR-0011, Decisions 2-3) - the
            # dashboard's threshold slider and red-team console read these
            # via the gateway rather than the frontend bundling stale data.
            # Missing at startup degrades to 404 on the /v1/eval/* routes,
            # not a crash: a fresh clone hasn't run `just eval` yet.
            eval_results_path=Path(
                os.environ.get("PORTCULLIS_EVAL_RESULTS_PATH", "docs/benchmarks/eval-results.json")
            ),
            eval_scores_path=Path(
                os.environ.get("PORTCULLIS_EVAL_SCORES_PATH", "docs/benchmarks/eval-scores.json")
            ),
            # The dashboard (Next.js dev server) is a different origin from
            # the gateway - CORS must name it explicitly rather than "*",
            # since the gateway also handles real request/response bodies.
            dashboard_origin=os.environ.get("PORTCULLIS_DASHBOARD_ORIGIN", "http://localhost:3000"),
        )
