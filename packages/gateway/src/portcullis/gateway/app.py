"""The gateway: an OpenAI-compatible transparent proxy plus the standalone
detection API (spec's API CONTRACT section), both backed by the same
DetectionPipeline (L0-L2, kNN, fusion, policy - M0-M5) and, when a caller
opts in with a `conversation_id`, L4's conversation state machine on top
(M7, ADR-0008) via `ConversationAwareDetector`.

`create_app(pipeline=..., conversation_detector=...)` lets a test inject
fakes and skip both real model loading and any storage backend entirely;
with neither given, the real ones build once at startup via `lifespan`,
matching `DetectionPipeline`'s own load-once-score-many contract
(pipeline.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from portcullis.core.l1 import Scope
from portcullis.core.l4 import InMemoryConversationStore
from portcullis.core.policy import Verdict

from .config import GatewayConfig
from .conversation import ConversationAwareDetector, embed_fn_for
from .egress import apply_scan_to_response, inject_canary, scan_and_relay_stream, scan_full_response
from .pipeline import DetectionPipeline, DetectionResult
from .proxy import UpstreamProxy
from .redis_store import RedisConversationStore
from .schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    BlockedErrorDetail,
    BlockedErrorResponse,
    ChatCompletionRequest,
    DetectRequest,
    DetectResponse,
    EgressBlockedErrorDetail,
    EgressBlockedErrorResponse,
    HealthResponse,
    LatencyBreakdownModel,
    MatchedRule,
    ReadyResponse,
    conversation_state_to_str,
    verdict_to_str,
)

_SCOPE_MAP = {
    "user": Scope.USER,
    "system": Scope.SYSTEM,
    "retrieved": Scope.RETRIEVED,
    "tool_result": Scope.TOOL_RESULT,
    "file": Scope.FILE,
}


def _to_detect_response(result: DetectionResult) -> DetectResponse:
    return DetectResponse(
        verdict=verdict_to_str(result.verdict),
        enforced=result.enforced,
        score=result.score,
        rationale=result.rationale,
        contributions=result.contributions,
        matched_rules=tuple(
            MatchedRule(
                rule_id=m.rule_id,
                name=m.name,
                severity=m.severity.name.lower(),
                labels=m.labels,
                weight=m.weight,
                span=m.span,
                matched_text=m.matched_text,
                rationale=m.rationale,
            )
            for m in result.matched_rules
        ),
        taxonomy_labels=result.taxonomy_labels,
        obfuscation_score=result.obfuscation_score,
        max_decode_depth=result.max_decode_depth,
        nearest_known_attack=result.nearest_attack_text,
        nearest_known_attack_family=result.nearest_attack_family,
        latency_ms=LatencyBreakdownModel(
            l0_ms=result.latency.l0_ms,
            l1_ms=result.latency.l1_ms,
            l2_ms=result.latency.l2_ms,
            knn_ms=result.latency.knn_ms,
            fusion_ms=result.latency.fusion_ms,
            total_ms=result.latency.total_ms,
        ),
        conversation_state=conversation_state_to_str(result.conversation_state),
    )


def get_conversation_detector(request: Request) -> ConversationAwareDetector:
    detector: ConversationAwareDetector | None = request.app.state.conversation_detector
    if detector is None:
        raise HTTPException(status_code=503, detail="detection pipeline not ready")
    return detector


def get_proxy(request: Request) -> UpstreamProxy:
    proxy: UpstreamProxy | None = request.app.state.proxy
    if proxy is None:
        raise HTTPException(status_code=503, detail="upstream proxy not ready")
    return proxy


def create_app(
    *,
    config: GatewayConfig | None = None,
    pipeline: DetectionPipeline | None = None,
    proxy: UpstreamProxy | None = None,
    conversation_detector: ConversationAwareDetector | None = None,
) -> FastAPI:
    """`pipeline`, `proxy` and `conversation_detector` let a test inject
    fakes/in-process transports and skip real model loading, real network
    sockets, and any storage backend entirely - see their own docstrings
    (pipeline.py, proxy.py, conversation.py)."""
    resolved_config = config or GatewayConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.pipeline is None:
            app.state.pipeline = DetectionPipeline.from_config(resolved_config)
        if app.state.proxy is None:
            app.state.proxy = UpstreamProxy(resolved_config.upstream_base_url)
        if app.state.conversation_detector is None:
            store = (
                RedisConversationStore.from_url(resolved_config.redis_url)
                if resolved_config.redis_url
                else InMemoryConversationStore()
            )
            app.state.conversation_detector = ConversationAwareDetector(
                app.state.pipeline,
                store,
                embed_fn_for(app.state.pipeline.embedder),
                ttl_s=resolved_config.conversation_ttl_s,
            )
        yield
        await app.state.proxy.aclose()

    app = FastAPI(title="PORTCULLIS gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = resolved_config
    app.state.pipeline = pipeline
    app.state.proxy = proxy
    app.state.conversation_detector = conversation_detector

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz(request: Request) -> ReadyResponse:
        if request.app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="detection pipeline not ready")
        return ReadyResponse(status="ready")

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(
        body: DetectRequest,
        conversation_dep: ConversationAwareDetector = Depends(get_conversation_detector),
    ) -> DetectResponse:
        result = conversation_dep.detect(
            body.text,
            conversation_id=body.conversation_id,
            scope=_SCOPE_MAP[body.scope],
            shadow=body.shadow,
        )
        return _to_detect_response(result)

    @app.post("/v1/detect/batch", response_model=BatchDetectResponse)
    async def detect_batch(
        body: BatchDetectRequest,
        conversation_dep: ConversationAwareDetector = Depends(get_conversation_detector),
    ) -> BatchDetectResponse:
        results = tuple(
            _to_detect_response(
                conversation_dep.detect(
                    item.text,
                    conversation_id=item.conversation_id,
                    scope=_SCOPE_MAP[item.scope],
                    shadow=item.shadow,
                )
            )
            for item in body.items
        )
        return BatchDetectResponse(results=results)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        conversation_dep: ConversationAwareDetector = Depends(get_conversation_detector),
        proxy_dep: UpstreamProxy = Depends(get_proxy),
    ) -> Response:
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        result = conversation_dep.detect(
            last_user,
            conversation_id=body.conversation_id,
            scope=Scope.USER,
            shadow=resolved_config.shadow_mode,
        )
        detect_response = _to_detect_response(result)

        if result.enforced and result.verdict is Verdict.BLOCK:
            return JSONResponse(
                status_code=400,
                content=BlockedErrorResponse(
                    error=BlockedErrorDetail(
                        message=f"Request blocked by PORTCULLIS: {result.rationale}",
                        portcullis=detect_response,
                    )
                ).model_dump(),
            )

        # conversation_id is a PORTCULLIS extension, not an OpenAI request
        # field - stripped before forwarding, so the upstream never sees it.
        payload = body.model_dump(exclude={"conversation_id"})
        headers = {
            "X-Portcullis-Verdict": verdict_to_str(result.verdict),
            "X-Portcullis-Score": f"{result.score:.6f}",
        }
        if result.conversation_state is not None:
            headers["X-Portcullis-Conversation-State"] = result.conversation_state.name.lower()

        # L5 egress inspection (M8, ADR-0009): a canary goes out in the
        # system prompt on every request, and this same request's response
        # is scanned for it plus secrets/exfiltration before the client
        # sees it - independent of the input-side verdict above.
        egress_payload, canary = inject_canary(payload)

        if not body.stream:
            upstream_response = await proxy_dep.complete(egress_payload)
            egress_result = scan_full_response(upstream_response, canary=canary)

            if egress_result.canary_leaked:
                return JSONResponse(
                    status_code=400,
                    content=EgressBlockedErrorResponse(
                        error=EgressBlockedErrorDetail(
                            message=(
                                "Response blocked by PORTCULLIS: confirmed system-prompt "
                                "extraction detected in the model's reply."
                            ),
                            rationale=egress_result.findings[0].rationale,
                        )
                    ).model_dump(),
                    headers=headers,
                )

            if egress_result.has_findings:
                upstream_response = apply_scan_to_response(upstream_response, egress_result)
            return JSONResponse(content=upstream_response, headers=headers)

        return StreamingResponse(
            scan_and_relay_stream(
                proxy_dep.stream(egress_payload), canary=canary, model=body.model
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    return app


# uvicorn's default module:attribute convention (`uvicorn portcullis.gateway.app:app`).
# Cheap to build eagerly: GatewayConfig.from_env() only reads env vars, and the real
# model load is deferred to `lifespan`, not done here.
app = create_app()
