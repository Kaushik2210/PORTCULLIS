"""A tiny deterministic fake LLM backend, standing in for a real provider so
the Milestone 6 base_url-swap demo needs no API key and costs nothing to run
(ADR-0007, Decision 2).

Implements the same `/v1/chat/completions` shape as the gateway's own proxy
target - streaming and non-streaming - as a genuinely separate FastAPI app
the gateway calls over real HTTP, not a function call standing in for one.
The reply is derived from the actual input (`"Mock response to: ..."`)
rather than a fixed string, so a demo run visibly proves the round trip
reflects what was actually sent.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .schemas import ChatCompletionRequest

app = FastAPI(title="portcullis-mock-upstream")

_TOKEN_DELAY_S = 0.01


def _reply_text(request: ChatCompletionRequest) -> str:
    last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    return f"Mock response to: {last_user[:200]}"


def _completion_id() -> str:
    return f"chatcmpl-mock-{uuid.uuid4().hex[:24]}"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> StreamingResponse | dict[str, object]:
    reply = _reply_text(request)
    completion_id = _completion_id()
    created = int(time.time())

    if not request.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                "completion_tokens": len(reply.split()),
                "total_tokens": 0,
            },
        }

    async def _stream() -> AsyncIterator[str]:
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(_TOKEN_DELAY_S)

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
