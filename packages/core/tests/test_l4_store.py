"""InMemoryConversationStore - the reference ConversationStore
implementation that lives in core (ADR-0008, Decision 4). TTL expiry is
tested against an injected clock, not real sleeping.
"""

from __future__ import annotations

from portcullis.core.l4 import ConversationRecord, ConversationState, InMemoryConversationStore


def test_get_before_any_set_is_none() -> None:
    store = InMemoryConversationStore()
    assert store.get("conv-1") is None


def test_set_then_get_round_trips() -> None:
    store = InMemoryConversationStore()
    record = ConversationRecord(state=ConversationState.PROBING, turn_count=3)
    store.set("conv-1", record, ttl_s=60.0)
    assert store.get("conv-1") == record


def test_entries_do_not_interfere_across_conversation_ids() -> None:
    store = InMemoryConversationStore()
    store.set("conv-1", ConversationRecord(turn_count=1), ttl_s=60.0)
    store.set("conv-2", ConversationRecord(turn_count=9), ttl_s=60.0)
    assert store.get("conv-1").turn_count == 1  # type: ignore[union-attr]
    assert store.get("conv-2").turn_count == 9  # type: ignore[union-attr]


def test_expired_entry_reads_as_none() -> None:
    now = [1000.0]
    store = InMemoryConversationStore(clock=lambda: now[0])
    store.set("conv-1", ConversationRecord(turn_count=1), ttl_s=10.0)
    now[0] = 1005.0
    assert store.get("conv-1") is not None
    now[0] = 1011.0
    assert store.get("conv-1") is None


def test_a_later_set_overwrites_the_earlier_record_and_ttl() -> None:
    now = [0.0]
    store = InMemoryConversationStore(clock=lambda: now[0])
    store.set("conv-1", ConversationRecord(turn_count=1), ttl_s=5.0)
    store.set("conv-1", ConversationRecord(turn_count=2), ttl_s=100.0)
    now[0] = 10.0  # would have expired the first TTL, not the second
    record = store.get("conv-1")
    assert record is not None
    assert record.turn_count == 2
