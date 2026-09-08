"""Wires L5 into the chat-completions proxy path (ADR-0009): injects a
canary into the outbound system message, then scans the response - full
body or SSE stream - for that canary plus secrets/exfiltration, applying
the differentiated action each finding kind gets (halt on canary, redact
and continue on everything else).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from portcullis.core.l5 import EgressScanResult, SlidingWindowScanner, generate_canary
from portcullis.core.l5 import inject_into_system_prompt as _inject
from portcullis.core.l5 import scan as _scan


def inject_canary(payload: dict[str, object]) -> tuple[dict[str, object], str]:
    """Returns a new payload (the original is left untouched) with the
    canary injected into the existing system message, or a new one added
    if the request had none - plus the canary itself, for scanning this
    same request's response."""
    canary = generate_canary()
    raw_messages = payload.get("messages", [])
    # pydantic's model_dump() preserves the field's own container type - a
    # `tuple[ChatMessage, ...]` field dumps as a tuple, not a list. Both
    # need handling here, or messages silently vanish (a bug this exact
    # line caused once already, caught by the gateway test suite).
    messages: list[dict[str, object]] = (
        [dict(m) for m in raw_messages if isinstance(m, dict)]
        if isinstance(raw_messages, list | tuple)
        else []
    )

    for message in messages:
        if message.get("role") == "system":
            message["content"] = _inject(str(message["content"]), canary)
            break
    else:
        messages.insert(0, {"role": "system", "content": _inject("", canary)})

    return {**payload, "messages": messages}, canary


def scan_full_response(response: dict[str, object], *, canary: str) -> EgressScanResult:
    """Non-streaming path: the whole reply already exists, so this is just
    `core.l5.scan` on the assistant's content - no sliding window needed."""
    content = _extract_content(response)
    return _scan(content, canary=canary)


def apply_scan_to_response(
    response: dict[str, object], result: EgressScanResult
) -> dict[str, object]:
    """Returns a new response dict with the assistant's content replaced
    by the redacted text. Callers check `result.canary_leaked` themselves
    first (ADR-0009, Decision 3) - a canary leak means the whole response
    is withheld, not redacted-and-returned, which this function doesn't
    decide on its own."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return response
    new_choices = [dict(c) for c in choices]
    message = dict(new_choices[0].get("message", {}))
    message["content"] = result.redacted_text
    new_choices[0]["message"] = message
    return {**response, "choices": new_choices}


def _extract_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content)


def _sse_chunk(content: str, *, model: str) -> bytes:
    payload = {
        "id": f"chatcmpl-portcullis-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_blocked_event(model: str) -> bytes:
    payload = {
        "id": f"chatcmpl-portcullis-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
        "portcullis_blocked": True,
        "portcullis_reason": (
            "A confirmed system-prompt extraction (canary leak) was detected mid-stream "
            "and the response was halted. Already-sent text cannot be recalled."
        ),
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


async def scan_and_relay_stream(
    chunks: AsyncIterator[bytes], *, canary: str, model: str
) -> AsyncIterator[bytes]:
    """Parses raw SSE bytes from the upstream (which may split an SSE
    event across chunks, or bundle several into one - `aiter_bytes()`
    makes no alignment guarantee), feeds each event's `delta.content`
    through a `SlidingWindowScanner`, and re-serialises whatever the
    scanner releases as new SSE chunks. Reconstructed chunks carry fresh
    ids rather than preserving the upstream's own - egress scanning
    already changes the content; byte-for-byte passthrough of chunk
    metadata isn't a guarantee this layer makes.
    """
    scanner = SlidingWindowScanner(canary=canary)
    buffer = b""

    async for raw in chunks:
        buffer += raw
        while b"\n\n" in buffer:
            line, buffer = buffer.split(b"\n\n", 1)
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[len("data: ") :]
            if data == "[DONE]":
                async for out in _flush(scanner, model=model, send_done=True):
                    yield out
                return

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            content = _delta_content(event)
            if not content:
                continue

            result = scanner.feed(content)
            if result.halted:
                yield _sse_blocked_event(model)
                return
            if result.release_text:
                yield _sse_chunk(result.release_text, model=model)

    async for out in _flush(scanner, model=model, send_done=True):
        yield out


async def _flush(
    scanner: SlidingWindowScanner, *, model: str, send_done: bool
) -> AsyncIterator[bytes]:
    final = scanner.finish()
    if final.halted:
        yield _sse_blocked_event(model)
        return
    if final.release_text:
        yield _sse_chunk(final.release_text, model=model)
    if send_done:
        yield b"data: [DONE]\n\n"


def _delta_content(event: dict[str, object]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
    content = delta.get("content", "") if isinstance(delta, dict) else ""
    return str(content) if content else ""


__all__ = [
    "apply_scan_to_response",
    "inject_canary",
    "scan_and_relay_stream",
    "scan_full_response",
]
