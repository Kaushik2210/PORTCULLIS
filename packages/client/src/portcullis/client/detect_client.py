"""A minimal typed client for the standalone detection API (`/v1/detect`,
`/v1/detect/batch`). Deliberately doesn't wrap `/v1/chat/completions` - the
entire point of that endpoint being OpenAI-compatible is that callers use
their existing OpenAI SDK against it, `base_url` swapped, not a
PORTCULLIS-specific client (spec's API CONTRACT section).
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

Scope = Literal["user", "system", "retrieved", "tool_result", "file"]


def _item(
    text: str, scope: Scope, shadow: bool, conversation_id: str | None
) -> dict[str, str | bool]:
    """`conversation_id` is only included when set, not sent as an explicit
    `null` - keeps the wire payload identical to pre-M7 callers who never
    pass it (server-side default is also omitted/None either way)."""
    item: dict[str, str | bool] = {"text": text, "scope": scope, "shadow": shadow}
    if conversation_id is not None:
        item["conversation_id"] = conversation_id
    return item


class MatchedRule(BaseModel):
    rule_id: str
    name: str
    severity: str
    labels: tuple[str, ...]
    weight: float
    span: tuple[int, int]
    matched_text: str
    rationale: str


class LatencyBreakdown(BaseModel):
    l0_ms: float
    l1_ms: float
    l2_ms: float
    knn_ms: float
    fusion_ms: float
    total_ms: float


class DetectResult(BaseModel):
    verdict: Literal["allow", "flag", "sanitise", "challenge", "block"]
    enforced: bool
    score: float
    rationale: str
    contributions: dict[str, float]
    matched_rules: tuple[MatchedRule, ...]
    taxonomy_labels: tuple[str, ...]
    obfuscation_score: float
    max_decode_depth: int
    nearest_known_attack: str | None
    nearest_known_attack_family: str | None
    latency_ms: LatencyBreakdown
    conversation_state: Literal["normal", "probing", "establishing", "exploiting"] | None = None


class DetectClient:
    """Synchronous client. A demo script or a CI red-team check calling
    this a few thousand times doesn't need an async client to do it well -
    the gateway itself is async where it matters (the SSE proxy path)."""

    def __init__(
        self,
        base_url: str = "",
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        """`client` lets a test pass a pre-built `httpx.Client` - including
        FastAPI's `TestClient`, which *is* an `httpx.Client` bridged to an
        in-process ASGI app - instead of opening a real socket. Ownership
        follows who built it: a client we constructed, we close; a client
        the caller handed us, the caller closes."""
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DetectClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def detect(
        self,
        text: str,
        *,
        scope: Scope = "user",
        shadow: bool = False,
        conversation_id: str | None = None,
    ) -> DetectResult:
        response = self._client.post("/v1/detect", json=_item(text, scope, shadow, conversation_id))
        response.raise_for_status()
        return DetectResult.model_validate(response.json())

    def detect_batch(
        self,
        texts: list[str],
        *,
        scope: Scope = "user",
        shadow: bool = False,
        conversation_id: str | None = None,
    ) -> list[DetectResult]:
        response = self._client.post(
            "/v1/detect/batch",
            json={"items": [_item(t, scope, shadow, conversation_id) for t in texts]},
        )
        response.raise_for_status()
        return [DetectResult.model_validate(r) for r in response.json()["results"]]
