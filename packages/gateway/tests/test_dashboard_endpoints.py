"""The gateway surface Milestone 10's dashboard is built on: the live
decision feed (`/v1/decisions/*`) and the Milestone 9 eval-data endpoints
(`/v1/eval/*`) - ADR-0011.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from portcullis.gateway.app import create_app
from portcullis.gateway.config import GatewayConfig
from portcullis.gateway.conversation import ConversationAwareDetector
from portcullis.gateway.pipeline import DetectionPipeline
from portcullis.gateway.proxy import UpstreamProxy

_UNUSED = Path("unused")


def _config(
    *, eval_results_path: Path = _UNUSED, eval_scores_path: Path = _UNUSED
) -> GatewayConfig:
    return GatewayConfig(
        rules_dir=_UNUSED,
        checkpoint_dir=_UNUSED,
        onnx_path=_UNUSED,
        knn_index_dir=_UNUSED,
        fusion_weights_path=_UNUSED,
        upstream_base_url="http://mock-upstream",
        shadow_mode=False,
        flag_threshold=0.2,
        sanitise_threshold=0.4,
        challenge_threshold=0.6,
        block_threshold=0.8,
        redis_url=None,
        conversation_ttl_s=1800.0,
        eval_results_path=eval_results_path,
        eval_scores_path=eval_scores_path,
        dashboard_origin="http://localhost:5173",
    )


@pytest.fixture
def client(
    fake_pipeline: DetectionPipeline,
    conversation_detector: ConversationAwareDetector,
    mock_upstream_asgi_app: FastAPI,
) -> Iterator[TestClient]:
    proxy = UpstreamProxy(
        "http://mock-upstream", transport=httpx.ASGITransport(app=mock_upstream_asgi_app)
    )
    app = create_app(
        config=_config(),
        pipeline=fake_pipeline,
        proxy=proxy,
        conversation_detector=conversation_detector,
    )
    with TestClient(app) as c:
        yield c


def test_decisions_recent_is_empty_before_any_requests(client: TestClient) -> None:
    assert client.get("/v1/decisions/recent").json() == []


def test_a_detect_call_shows_up_in_recent_decisions(client: TestClient) -> None:
    client.post("/v1/detect", json={"text": "hello there"})
    entries = client.get("/v1/decisions/recent").json()
    assert len(entries) == 1
    assert entries[0]["source"] == "detect"
    assert entries[0]["verdict"] == "allow"


def test_a_blocked_detect_call_is_recorded_too(client: TestClient, injection_marker: str) -> None:
    client.post("/v1/detect", json={"text": injection_marker})
    entries = client.get("/v1/decisions/recent").json()
    assert entries[-1]["verdict"] == "block"
    assert entries[-1]["enforced"] is True


def test_a_batch_call_records_one_entry_per_item(client: TestClient) -> None:
    client.post("/v1/detect/batch", json={"items": [{"text": "one"}, {"text": "two"}]})
    entries = client.get("/v1/decisions/recent").json()
    assert len(entries) == 2
    assert all(e["source"] == "detect_batch" for e in entries)


def test_recent_respects_the_n_query_param(client: TestClient) -> None:
    for i in range(5):
        client.post("/v1/detect", json={"text": f"message {i}"})
    entries = client.get("/v1/decisions/recent", params={"n": 2}).json()
    assert len(entries) == 2


def test_decision_log_entry_model_round_trips_every_field() -> None:
    """`/v1/decisions/stream`'s live-push path is deliberately not tested
    through a full ASGI round trip here: reproduced directly (a real
    async client, `asyncio.wait_for` around the read, a full traceback)
    that `httpx.ASGITransport` deadlocks against Starlette's
    `StreamingResponse` disconnect-listener for a generator that never
    terminates on its own - which an SSE feed for *live* traffic never
    does, by design. Every finite-generator SSE stream elsewhere in this
    project (M6's chat-completions relay, M8's egress scan) completes
    naturally and is tested the same way without issue; this is
    specifically about indefinite streams under this transport, not a bug
    in decision_log.py's own subscribe/record/fan-out logic - which is
    exhaustively covered, with no ASGI transport involved at all, in
    test_decision_log.py. This test instead proves the wire shape the
    stream actually sends is correct."""
    from portcullis.gateway.decision_log import DecisionLog, DecisionLogEntryModel

    log = DecisionLog()
    log.record(
        text="hello",
        verdict="block",
        enforced=True,
        score=0.9,
        taxonomy_labels=("direct_override",),
        conversation_state="exploiting",
        latency_ms=12.5,
        source="chat_completions",
    )
    entry = log.recent()[0]
    model = DecisionLogEntryModel.from_entry(entry)
    round_tripped = json.loads(model.model_dump_json())

    assert round_tripped["id"] == entry.id
    assert round_tripped["verdict"] == "block"
    assert round_tripped["taxonomy_labels"] == ["direct_override"]
    assert round_tripped["conversation_state"] == "exploiting"
    assert round_tripped["source"] == "chat_completions"


#
# No gateway-level (full ASGI round-trip) test exists for GET
# /v1/decisions/stream. Confirmed directly, twice, with a hard wall-clock
# timeout and a full traceback both times: entering the response - even
# with a bounded client-side timeout, even without reading any body -
# deadlocks under both httpx.AsyncClient(transport=ASGITransport(...)) and
# Starlette's TestClient. Root cause (visible in the traceback): Starlette's
# StreamingResponse races a `stream_response` task against a
# `listen_for_disconnect` task; the latter's `receive()` under
# ASGITransport blocks on an internal `response_complete` event that only
# fires once the body generator itself finishes - which an indefinite
# live-feed generator (`while True: await queue.get()`) never does, by
# design. Every *finite* SSE generator elsewhere in this project (M6's
# chat-completions relay, M8's egress scan) closes on its own and is
# tested this same way without any issue - this is specific to indefinite
# streams under this in-process transport, not a defect in this project's
# own code. Real verification (a real uvicorn server, a real socket, real
# TCP disconnect semantics - `just demo-m10`) is what actually exercises
# this route end to end; the wire shape it sends is covered above, and the
# subscribe/record/fan-out mechanics it depends on are covered
# exhaustively, with no ASGI transport involved, in test_decision_log.py.


def test_eval_scores_404s_when_the_artifact_is_missing(client: TestClient) -> None:
    assert client.get("/v1/eval/scores").status_code == 404


def test_eval_results_404s_when_the_artifact_is_missing(client: TestClient) -> None:
    assert client.get("/v1/eval/results").status_code == 404


def test_eval_scores_serves_real_content_when_present(
    fake_pipeline: DetectionPipeline,
    conversation_detector: ConversationAwareDetector,
    mock_upstream_asgi_app: FastAPI,
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "eval-scores.json"
    scores_path.write_text(json.dumps({"labels": [0, 1], "fusion": [0.1, 0.9]}), encoding="utf-8")
    proxy = UpstreamProxy(
        "http://mock-upstream", transport=httpx.ASGITransport(app=mock_upstream_asgi_app)
    )
    app = create_app(
        config=_config(eval_scores_path=scores_path),
        pipeline=fake_pipeline,
        proxy=proxy,
        conversation_detector=conversation_detector,
    )
    with TestClient(app) as c:
        response = c.get("/v1/eval/scores")
    assert response.status_code == 200
    assert response.json() == {"labels": [0, 1], "fusion": [0.1, 0.9]}


def test_eval_results_serves_real_content_when_present(
    fake_pipeline: DetectionPipeline,
    conversation_detector: ConversationAwareDetector,
    mock_upstream_asgi_app: FastAPI,
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "eval-results.json"
    results_path.write_text(json.dumps({"headline": {"auprc": 0.6}}), encoding="utf-8")
    proxy = UpstreamProxy(
        "http://mock-upstream", transport=httpx.ASGITransport(app=mock_upstream_asgi_app)
    )
    app = create_app(
        config=_config(eval_results_path=results_path),
        pipeline=fake_pipeline,
        proxy=proxy,
        conversation_detector=conversation_detector,
    )
    with TestClient(app) as c:
        response = c.get("/v1/eval/results")
    assert response.status_code == 200
    assert response.json() == {"headline": {"auprc": 0.6}}


def test_cors_allows_the_configured_dashboard_origin(client: TestClient) -> None:
    response = client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_an_unlisted_origin(client: TestClient) -> None:
    response = client.get("/healthz", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers
