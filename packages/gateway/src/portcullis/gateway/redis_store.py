"""The Redis-backed `ConversationStore` (ADR-0008, Decision 4). Sync
`redis.Redis`, not `redis.asyncio` - matches `ConversationStore`'s sync
Protocol exactly, and is consistent with `L2Scorer`/`KnnScorer` already
making blocking calls from inside async route handlers (M6): a Redis round
trip is a similar order of magnitude to those, and threading a second async
call convention through just for this would be a bigger change than this
milestone's scope, revisitable if Milestone 9's load testing shows it
matters.

`redis` (the client library) was not named separately from "Redis" (the
service) in the spec's stack list - the same category of gap `httpx`/
`uvicorn` were at Milestone 6, treated as implied by the already-approved
entry rather than re-asked, and disclosed here rather than added quietly.
"""

from __future__ import annotations

import json

import redis

from portcullis.core.l4 import ConversationRecord, ConversationState
from portcullis.core.policy import Verdict

_KEY_PREFIX = "portcullis:conversation:"


def _to_json(record: ConversationRecord) -> str:
    return json.dumps(
        {
            "state": record.state.value,
            "turn_count": record.turn_count,
            "cumulative_risk": record.cumulative_risk,
            "last_turn_at": record.last_turn_at,
            "established_labels": sorted(record.established_labels),
            "recent_scores": list(record.recent_scores),
            "last_verdict": record.last_verdict.value,
            "last_embedding": list(record.last_embedding),
        }
    )


def _from_json(raw: str) -> ConversationRecord:
    data = json.loads(raw)
    return ConversationRecord(
        state=ConversationState(data["state"]),
        turn_count=data["turn_count"],
        cumulative_risk=data["cumulative_risk"],
        last_turn_at=data["last_turn_at"],
        established_labels=frozenset(data["established_labels"]),
        recent_scores=tuple(data["recent_scores"]),
        last_verdict=Verdict(data["last_verdict"]),
        last_embedding=tuple(data["last_embedding"]),
    )


class RedisConversationStore:
    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> RedisConversationStore:
        return cls(redis.Redis.from_url(url, decode_responses=True))

    def get(self, conversation_id: str) -> ConversationRecord | None:
        raw = self._client.get(_KEY_PREFIX + conversation_id)
        if raw is None:
            return None
        # decode_responses=True (from_url) means this is always str at
        # runtime; the untyped client's own stub doesn't encode that.
        return _from_json(raw if isinstance(raw, str) else raw.decode())

    def set(self, conversation_id: str, record: ConversationRecord, *, ttl_s: float) -> None:
        self._client.set(_KEY_PREFIX + conversation_id, _to_json(record), ex=int(ttl_s))
