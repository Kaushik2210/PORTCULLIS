"""Where a `ConversationRecord` lives between requests.

`ConversationStore` is a `Protocol`, not an ABC with a Redis import - core
must stay as dependency-light as L0/L1/fusion/policy already are (ADR-0001's
"never import torch" extends to "never import an infra client" here,
ADR-0008 Decision 4). `InMemoryConversationStore` is the one concrete
implementation that belongs in core: it is pure Python, no infra, and every
L4 unit test and the gateway's fake-pipeline suite use it so `just test`
never touches Docker. The Redis-backed implementation lives in
`packages/gateway` (`redis_store.py`), where the actual infra dependency is.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from .types import ConversationRecord


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> ConversationRecord | None: ...

    def set(self, conversation_id: str, record: ConversationRecord, *, ttl_s: float) -> None: ...


class InMemoryConversationStore:
    """A dict with manual TTL eviction - checked lazily on read, not with a
    background sweep. Fine for tests and for a single-process gateway; a
    multi-process deployment needs the Redis-backed store precisely because
    this one's state is process-local."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        """`clock` lets a test control TTL expiry without real sleeping -
        the same seam `advance()` gets via its own `now` parameter."""
        self._records: dict[str, tuple[ConversationRecord, float]] = {}
        self._clock = clock

    def get(self, conversation_id: str) -> ConversationRecord | None:
        entry = self._records.get(conversation_id)
        if entry is None:
            return None
        record, expires_at = entry
        if self._clock() >= expires_at:
            del self._records[conversation_id]
            return None
        return record

    def set(self, conversation_id: str, record: ConversationRecord, *, ttl_s: float) -> None:
        self._records[conversation_id] = (record, self._clock() + ttl_s)
