"""DecisionLog - the bounded history + live fan-out the dashboard's live
traffic feed is built on (ADR-0011, Decision 1).
"""

from __future__ import annotations

import asyncio

import pytest

from portcullis.gateway.decision_log import DecisionLog


def _record(log: DecisionLog, text: str = "hello", verdict: str = "allow") -> None:
    log.record(
        text=text,
        verdict=verdict,
        enforced=True,
        score=0.1,
        taxonomy_labels=(),
        conversation_state=None,
        latency_ms=1.0,
        source="detect",
    )


def test_recent_is_empty_before_anything_is_recorded() -> None:
    assert DecisionLog().recent() == ()


def test_recorded_entries_come_back_in_order() -> None:
    log = DecisionLog()
    _record(log, text="first")
    _record(log, text="second")
    entries = log.recent()
    assert [e.text_preview for e in entries] == ["first", "second"]


def test_recent_respects_n() -> None:
    log = DecisionLog()
    for i in range(10):
        _record(log, text=str(i))
    assert len(log.recent(n=3)) == 3
    assert [e.text_preview for e in log.recent(n=3)] == ["7", "8", "9"]


def test_history_is_bounded() -> None:
    log = DecisionLog()
    for i in range(600):
        _record(log, text=str(i))
    assert len(log.recent(n=1000)) == 500
    assert log.recent(n=1)[0].text_preview == "599"


def test_long_text_is_truncated_to_a_preview() -> None:
    log = DecisionLog()
    _record(log, text="x" * 1000)
    assert len(log.recent()[0].text_preview) == 200


def test_each_entry_gets_a_unique_id() -> None:
    log = DecisionLog()
    _record(log)
    _record(log)
    ids = {e.id for e in log.recent()}
    assert len(ids) == 2


@pytest.mark.anyio
async def test_a_subscriber_receives_new_entries() -> None:
    log = DecisionLog()
    queue = log.subscribe()
    _record(log, text="live one")
    entry = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert entry.text_preview == "live one"


@pytest.mark.anyio
async def test_unsubscribing_stops_further_delivery() -> None:
    log = DecisionLog()
    queue = log.subscribe()
    log.unsubscribe(queue)
    _record(log, text="after unsubscribe")
    assert queue.empty()


@pytest.mark.anyio
async def test_multiple_subscribers_each_get_their_own_copy() -> None:
    log = DecisionLog()
    q1 = log.subscribe()
    q2 = log.subscribe()
    _record(log, text="broadcast")
    e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1.text_preview == e2.text_preview == "broadcast"
