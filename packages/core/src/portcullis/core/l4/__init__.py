"""L4 - conversation state machine. Catches what stateless detection
structurally cannot: attacks assembled across turns rather than sitting
inside any single one (ADR-0008).

`advance()` folds one turn into a `ConversationRecord`; `apply_floor()`
turns the resulting state into a verdict floor applied on top of fusion's
own verdict, not a fifth fusion input (ADR-0008, Decision 1). Storage is a
`Protocol` (`ConversationStore`) so this package stays as dependency-light
as L0/L1/fusion/policy - `InMemoryConversationStore` is the one concrete
backend that belongs here; the Redis-backed one lives in packages/gateway.
"""

from .machine import DEFAULT_HALF_LIFE_S, advance, apply_floor, verdict_floor
from .store import ConversationStore, InMemoryConversationStore
from .types import ConversationRecord, ConversationState, TurnSignal

__all__ = [
    "DEFAULT_HALF_LIFE_S",
    "ConversationRecord",
    "ConversationState",
    "ConversationStore",
    "InMemoryConversationStore",
    "TurnSignal",
    "advance",
    "apply_floor",
    "verdict_floor",
]
