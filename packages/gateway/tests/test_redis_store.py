"""RedisConversationStore against a real Redis - `needs_infra`, not part of
`just check` (a fresh clone has no Docker services running). Run with
`just up` first, then `just test-infra`.

Uses DB 15 (Redis's convention for "not DB 0") and this project's own key
prefix, so this suite is safe to run against a shared local Redis that also
has other applications' data on DB 0 - as it happens, this dev machine
does (a different project's Redis container already bound to port 6379;
see ADR-0008's environment note). Every key this test creates is cleaned
up in a fixture teardown, not left behind.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest
import redis

from portcullis.core.l4 import ConversationRecord, ConversationState
from portcullis.core.policy import Verdict
from portcullis.gateway.redis_store import RedisConversationStore

pytestmark = pytest.mark.needs_infra

_TEST_REDIS_URL = os.environ.get("PORTCULLIS_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def store() -> Iterator[RedisConversationStore]:
    client = redis.Redis.from_url(_TEST_REDIS_URL, decode_responses=True)
    client.ping()  # fail fast with a clear error if Redis isn't reachable
    yield RedisConversationStore(client)
    for key in client.scan_iter("portcullis:conversation:test-*"):
        client.delete(key)


def test_get_before_any_set_is_none(store: RedisConversationStore) -> None:
    assert store.get("test-conv-1") is None


def test_set_then_get_round_trips_every_field(store: RedisConversationStore) -> None:
    record = ConversationRecord(
        state=ConversationState.ESTABLISHING,
        turn_count=3,
        cumulative_risk=0.42,
        last_turn_at=1_700_000_000.0,
        established_labels=frozenset({"role_play_jailbreak", "direct_override"}),
        recent_scores=(0.1, 0.2, 0.3),
        last_verdict=Verdict.FLAG,
        last_embedding=(0.1, 0.2, 0.3, 0.4),
    )
    store.set("test-conv-1", record, ttl_s=60.0)
    assert store.get("test-conv-1") == record


def test_ttl_actually_expires_the_key(store: RedisConversationStore) -> None:
    store.set("test-conv-ttl", ConversationRecord(turn_count=1), ttl_s=1.0)
    assert store.get("test-conv-ttl") is not None
    time.sleep(1.5)
    assert store.get("test-conv-ttl") is None
