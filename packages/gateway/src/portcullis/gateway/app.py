"""The gateway: an OpenAI-compatible transparent proxy plus the standalone
detection API (spec's API CONTRACT section), both backed by the same
DetectionPipeline (L0-L2, kNN, fusion, policy - M0-M5).

`create_app(pipeline=...)` lets a test inject a fake pipeline and skip
loading a real ONNX/sentence-transformer checkpoint entirely; with no
pipeline given, the real one loads once at startup via `lifespan`, matching
DetectionPipeline's own load-once-score-many contract (pipeline.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from portcullis.core.l1 import Scope
from portcullis.core.policy import Verdict

from .config import GatewayConfig
from .pipeline import DetectionPipeline, DetectionResult
from .proxy import UpstreamProxy
from .schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    BlockedErrorDetail,
    BlockedErrorResponse,
    ChatCompletionRequest,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    LatencyBreakdownModel,
    MatchedRule,
    ReadyResponse,
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
    )


def get_pipeline(request: Request) -> DetectionPipeline:
    pipeline: DetectionPipeline | None = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="detection pipeline not ready")
    return pipeline


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
) -> FastAPI:
    """`pipeline` and `proxy` let a test inject fakes/in-process transports
    and skip both real model loading and real network sockets entirely -
    see their own docstrings (pipeline.py, proxy.py)."""
    resolved_config = config or GatewayConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.pipeline is None:
            app.state.pipeline = DetectionPipeline.from_config(resolved_config)
        if app.state.proxy is None:
            app.state.proxy = UpstreamProxy(resolved_config.upstream_base_url)
        yield
        await app.state.proxy.aclose()

    app = FastAPI(title="PORTCULLIS gateway", version="0.1.0", lifespan=lifespan)
    app.state.config = resolved_config
    app.state.pipeline = pipeline
    app.state.proxy = proxy

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
        body: DetectRequest, pipeline_dep: DetectionPipeline = Depends(get_pipeline)
    ) -> DetectResponse:
        result = pipeline_dep.detect(body.text, scope=_SCOPE_MAP[body.scope], shadow=body.shadow)
        return _to_detect_response(result)

    @app.post("/v1/detect/batch", response_model=BatchDetectResponse)
    async def detect_batch(
        body: BatchDetectRequest, pipeline_dep: DetectionPipeline = Depends(get_pipeline)
    ) -> BatchDetectResponse:
        results = tuple(
            _to_detect_response(
                pipeline_dep.detect(item.text, scope=_SCOPE_MAP[item.scope], shadow=item.shadow)
            )
            for item in body.items
        )
        return BatchDetectResponse(results=results)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        pipeline_dep: DetectionPipeline = Depends(get_pipeline),
        proxy_dep: UpstreamProxy = Depends(get_proxy),
    ) -> Response:
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        result = pipeline_dep.detect(
            last_user, scope=Scope.USER, shadow=resolved_config.shadow_mode
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

        payload = body.model_dump()
        headers = {
            "X-Portcullis-Verdict": verdict_to_str(result.verdict),
            "X-Portcullis-Score": f"{result.score:.6f}",
        }

        if not body.stream:
            upstream_response = await proxy_dep.complete(payload)
            return JSONResponse(content=upstream_response, headers=headers)

        return StreamingResponse(
            proxy_dep.stream(payload), media_type="text/event-stream", headers=headers
        )

    return app


# uvicorn's default module:attribute convention (`uvicorn portcullis.gateway.app:app`).
# Cheap to build eagerly: GatewayConfig.from_env() only reads env vars, and the real
# model load is deferred to `lifespan`, not done here.
app = create_app()
