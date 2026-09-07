"""Relays a chat-completion request to whatever upstream is configured
(the mock by default; a real provider is a `base_url` config change,
ADR-0007) over real HTTP via httpx - streaming and non-streaming both.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class UpstreamProxy:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """`transport` lets a test point this at an in-process ASGI app
        (`httpx.ASGITransport(app=mock_upstream.app)`) instead of a real
        socket - the same proxy code path either way, no network required
        for the test suite to exercise it."""
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, payload: dict[str, object]) -> dict[str, object]:
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        result: dict[str, object] = response.json()
        return result

    async def stream(self, payload: dict[str, object]) -> AsyncIterator[bytes]:
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
