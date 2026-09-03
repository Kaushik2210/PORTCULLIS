"""Rule types for the L1 signature engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..l0 import ViewKind


class Severity(Enum):
    """How much a match should count against the request.

    Ordered, so policy can compare. Severity is a rule-author judgement about
    the *pattern*; it is not a probability and is not the final verdict - that
    comes from fusion (ADR-0001).
    """

    INFO = 10
    LOW = 20
    MEDIUM = 30
    HIGH = 40
    CRITICAL = 50

    def __lt__(self, other: Severity) -> bool:
        return self.value < other.value


class Scope(Enum):
    """Where a span of text came from.

    The same sentence means different things from different sources. "Ignore
    previous instructions" typed by a user is a request; the identical string
    inside a retrieved PDF is an attack, because no human chose to put it in
    front of the model. Scope is how a rule expresses that distinction.
    """

    USER = "user"
    SYSTEM = "system"
    RETRIEVED = "retrieved"
    TOOL_RESULT = "tool_result"
    FILE = "file"


# A match found in a lossy view is real evidence, but weaker evidence: leet
# folding turns '5' into 's' and could have rewritten something innocent.
# Discounting here rather than in fusion keeps the reason legible in the trace.
VIEW_CONFIDENCE: dict[ViewKind, float] = {
    ViewKind.CANONICAL: 1.0,
    ViewKind.DECODED: 0.9,
    ViewKind.FOLDED: 0.6,
}


@dataclass(frozen=True, slots=True)
class RuleTests:
    """A rule's own worked examples.

    Both directions are mandatory. A rule that cannot name what it must not
    match cannot be reviewed, and unfalsifiable rules are how a signature set
    accumulates a false-positive rate nobody can attribute.
    """

    positive: tuple[str, ...]
    negative: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    name: str
    description: str
    severity: Severity
    labels: tuple[str, ...]
    scope: tuple[Scope, ...]
    views: tuple[ViewKind, ...]
    weight: float
    matcher: MatchNode
    tests: RuleTests
    references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """One firing, carrying everything needed to explain itself.

    `span` is in the coordinates of the *original* user text, translated back
    through L0's offset maps. A span in normalised coordinates would be
    meaningless to whoever reads the decision.
    """

    rule_id: str
    name: str
    severity: Severity
    labels: tuple[str, ...]
    weight: float
    view: ViewKind
    span: tuple[int, int]
    matched_text: str
    rationale: str


@dataclass(frozen=True, slots=True)
class L1Result:
    matches: tuple[RuleMatch, ...]
    score: float

    @property
    def fired(self) -> bool:
        return bool(self.matches)


# Imported late to avoid a circular reference at module import time.
from .dsl import MatchNode  # noqa: E402
