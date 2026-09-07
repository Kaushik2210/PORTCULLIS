"""Routing, blocking and proxying behaviour, against the fake pipeline
(conftest.py) and an in-process mock upstream reached via ASGITransport -
no real model, no real socket, so this suite runs in the default `just
test` pass rather than needing `needs_model`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from portcullis.gateway.app import create_app
from portcullis.gateway.config import GatewayConfig
from portcullis.gateway.pipeline import DetectionPipeline
from portcullis.gateway.proxy import UpstreamProxy

_UNUSED = Path("unused")  # never read: the fake pipeline skips real loading


def _config() -> GatewayConfig:
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
    )


@pytest.fixture
def client(
    fake_pipeline: DetectionPipeline, mock_upstream_asgi_app: FastAPI
) -> Iterator[TestClient]:
    proxy = UpstreamProxy(
        "http://mock-upstream",
        transport=httpx.ASGITransport(app=mock_upstream_asgi_app),
    )
    app = create_app(config=_config(), pipeline=fake_pipeline, proxy=proxy)
    with TestClient(app) as c:
        yield c


def test_healthz_does_not_need_a_pipeline() -> None:
    # No `with`: lifespan never runs, so this proves /healthz answers even
    # before (or without) the real pipeline ever loading - liveness, not
    # readiness. Entering the lifespan context here would trigger real
    # model loading (pipeline=None) and defeat the point of the test.
    app = create_app(config=_config(), pipeline=None, proxy=None)
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_is_503_without_a_pipeline() -> None:
    app = create_app(config=_config(), pipeline=None, proxy=None)
    response = TestClient(app).get("/readyz")  # no `with`: lifespan never runs
    assert response.status_code == 503


def test_readyz_is_ready_with_a_pipeline(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_detect_allows_benign_text(client: TestClient) -> None:
    response = client.post("/v1/detect", json={"text": "please summarise this document"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "allow"
    assert body["enforced"] is True
    assert body["matched_rules"] == []


def test_detect_blocks_injection_text(client: TestClient, injection_marker: str) -> None:
    response = client.post("/v1/detect", json={"text": injection_marker})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "block"
    assert len(body["matched_rules"]) == 1
    assert body["matched_rules"][0]["rule_id"] == "fake-001"
    assert "direct_override" in body["taxonomy_labels"]


def test_detect_shadow_mode_is_not_enforced(client: TestClient, injection_marker: str) -> None:
    response = client.post("/v1/detect", json={"text": injection_marker, "shadow": True})
    body = response.json()
    assert body["verdict"] == "block"
    assert body["enforced"] is False


def test_detect_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/v1/detect", json={"text": ""})
    assert response.status_code == 422


def test_detect_batch(client: TestClient, injection_marker: str) -> None:
    response = client.post(
        "/v1/detect/batch",
        json={"items": [{"text": "hello there"}, {"text": injection_marker}]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["verdict"] for r in results] == ["allow", "block"]


def test_chat_completions_proxies_allowed_traffic(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "hello there"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert "hello there" in response.json()["choices"][0]["message"]["content"]
    assert response.headers["x-portcullis-verdict"] == "allow"


def test_chat_completions_blocks_injection_before_reaching_upstream(
    client: TestClient, injection_marker: str
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": injection_marker}],
            "stream": False,
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "portcullis_block"
    assert body["error"]["portcullis"]["verdict"] == "block"


def test_chat_completions_streams_via_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "hello there"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "data: [DONE]" in body
    assert "hello" in body
