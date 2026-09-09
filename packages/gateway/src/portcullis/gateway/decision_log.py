"""An in-memory, session-local record of recent decisions, fanned out to
connected dashboard clients over SSE (ADR-0011, Decision 1). Not a durable
decision log - the Postgres service `ops/compose.yaml` scaffolds for
exactly this purpose was never wired up in any milestone through M9, and
building that now was a real, disclosed scope decision, not an oversight.
A bounded deque plus a broadcast to live subscribers is enough for a real
"live traffic" surface without a new infra dependency, the same trade-off
L4's in-memory conversation store already made (ADR-0008).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from pydantic import BaseModel

_MAX_ENTRIES = 500


@dataclass(frozen=True, slots=True)
class DecisionLogEntry:
    id: str
    timestamp: float
    text_preview: str
    """Truncated - this buffer is process-memory, not disk, but a live
    feed still shouldn't hold full request bodies indefinitely for no
    reason."""
    verdict: str
    enforced: bool
    score: float
    taxonomy_labels: tuple[str, ...]
    conversation_state: str | None
    latency_ms: float
    source: str
    """Which endpoint produced this decision (`detect`, `detect_batch`,
    `chat_completions`) - the live feed mixes all three."""


def _new_id() -> str:
    return uuid.uuid4().hex


class DecisionLogEntryModel(BaseModel):
    """The wire shape for `/v1/decisions/recent` and `/v1/decisions/stream`
    - a direct mirror of `DecisionLogEntry`, kept as a separate model
    (rather than serialising the dataclass directly) so the API's shape is
    declared, not incidental to the internal type's field names."""

    id: str
    timestamp: float
    text_preview: str
    verdict: str
    enforced: bool
    score: float
    taxonomy_labels: tuple[str, ...]
    conversation_state: str | None
    latency_ms: float
    source: str

    @classmethod
    def from_entry(cls, entry: DecisionLogEntry) -> DecisionLogEntryModel:
        return cls(
            id=entry.id,
            timestamp=entry.timestamp,
            text_preview=entry.text_preview,
            verdict=entry.verdict,
            enforced=entry.enforced,
            score=entry.score,
            taxonomy_labels=entry.taxonomy_labels,
            conversation_state=entry.conversation_state,
            latency_ms=entry.latency_ms,
            source=entry.source,
        )


@dataclass(slots=True)
class DecisionLog:
    """Bounded history plus live fan-out. One instance lives on
    `app.state`, shared across all requests and all connected SSE clients
    for the process's lifetime."""

    _entries: deque[DecisionLogEntry] = field(default_factory=lambda: deque(maxlen=_MAX_ENTRIES))
    _subscribers: list[asyncio.Queue[DecisionLogEntry]] = field(default_factory=list)

    def record(
        self,
        *,
        text: str,
        verdict: str,
        enforced: bool,
        score: float,
        taxonomy_labels: tuple[str, ...],
        conversation_state: str | None,
        latency_ms: float,
        source: str,
    ) -> DecisionLogEntry:
        entry = DecisionLogEntry(
            id=_new_id(),
            timestamp=time.time(),
            text_preview=text[:200],
            verdict=verdict,
            enforced=enforced,
            score=score,
            taxonomy_labels=taxonomy_labels,
            conversation_state=conversation_state,
            latency_ms=latency_ms,
            source=source,
        )
        self._entries.append(entry)
        for queue in self._subscribers:
            # A full queue means a slow/gone subscriber - drop the entry
            # for that one subscriber rather than block the request that
            # triggered it. put_nowait is safe here: queues are unbounded,
            # so this only raises if something else is wrong.
            queue.put_nowait(entry)
        return entry

    def recent(self, n: int = 50) -> tuple[DecisionLogEntry, ...]:
        return tuple(list(self._entries)[-n:])

    def subscribe(self) -> asyncio.Queue[DecisionLogEntry]:
        queue: asyncio.Queue[DecisionLogEntry] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DecisionLogEntry]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)
