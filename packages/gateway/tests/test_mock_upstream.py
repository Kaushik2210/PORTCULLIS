"""The mock upstream is its own small FastAPI app (ADR-0007) - tested in
isolation from the gateway so a broken reply format is caught here, not
buried inside a gateway-proxy test failure.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from portcullis.gateway.mock_upstream import app


def _reconstruct_streamed_content(sse_body: str) -> str:
    """Each SSE chunk carries one word in its own JSON blob (mock_upstream's
    token-by-token simulation) - the full reply only exists once every
    chunk's `delta.content` is extracted and joined, not as a literal
    substring of the raw SSE text."""
    parts: list[str] = []
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line.removeprefix("data: "))
        parts.append(chunk["choices"][0]["delta"].get("content", ""))
    return "".join(parts)


def test_healthz() -> None:
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200


def test_non_streaming_reply_echoes_input() -> None:
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "what is the weather"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert "what is the weather" in body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"


def test_streaming_reply_ends_with_done_marker() -> None:
    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "stream this back"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "stream this back" in _reconstruct_streamed_content(body)
    assert body.rstrip().endswith("data: [DONE]")


def test_reply_uses_the_last_user_message() -> None:
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
            "stream": False,
        },
    )
    content = response.json()["choices"][0]["message"]["content"]
    assert "second question" in content
    assert "first question" not in content
