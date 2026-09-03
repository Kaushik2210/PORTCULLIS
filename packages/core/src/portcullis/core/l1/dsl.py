"""The rule match DSL.

Deliberately tiny: `regex`, `all`, `any`, `not`, `near`. Every operator earns
its place by expressing something rules actually need and regex alone expresses
badly.

`near` is the one worth defending. Instruction-override attacks are
characterised by two ideas appearing close together - a negation verb and a
reference to prior instructions - in an order and phrasing that varies
endlessly. A single regex covering the variations is unreadable and brittle;
two simple patterns plus a distance bound is neither.

Patterns are compiled and screened at load time. There is no runtime timeout in
Python's `re`, so a catastrophically backtracking pattern would be a DoS in the
request path - the screen below rejects the nested-quantifier shapes that cause
it before a rule can ship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


class RuleSyntaxError(ValueError):
    """A ruleset failed to load. Always names the offending rule."""


# Nested quantifiers over a group - (a+)+, (a*)* , (.+)* - are the classic
# catastrophic-backtracking shape. This is a screen, not a proof; it catches
# the accidental cases that arise from ordinary rule authoring.
_REDOS = re.compile(r"\([^)]*[+*]\)[+*{]")

MAX_PATTERN_LEN = 400


class MatchNode(Protocol):
    """A node in a compiled match expression."""

    def find(self, text: str) -> tuple[int, int] | None:
        """Return the span this node matches, or None."""
        ...


@dataclass(frozen=True, slots=True)
class RegexNode:
    pattern: re.Pattern[str]

    def find(self, text: str) -> tuple[int, int] | None:
        m = self.pattern.search(text)
        return m.span() if m else None


@dataclass(frozen=True, slots=True)
class AllNode:
    children: tuple[MatchNode, ...]

    def find(self, text: str) -> tuple[int, int] | None:
        spans = [c.find(text) for c in self.children]
        if any(s is None for s in spans):
            return None
        real = [s for s in spans if s is not None]
        # Report the enclosing extent: the evidence is the conjunction, so
        # highlighting only one conjunct would under-explain the decision.
        return (min(s[0] for s in real), max(s[1] for s in real))


@dataclass(frozen=True, slots=True)
class AnyNode:
    children: tuple[MatchNode, ...]

    def find(self, text: str) -> tuple[int, int] | None:
        for child in self.children:
            span = child.find(text)
            if span is not None:
                return span
        return None


@dataclass(frozen=True, slots=True)
class NotNode:
    child: MatchNode

    def find(self, text: str) -> tuple[int, int] | None:
        # A negation matches "everywhere" when satisfied. It carries no span of
        # its own, so it returns a zero-width one and relies on being combined
        # under an `all` with something that does.
        return None if self.child.find(text) is not None else (0, 0)


@dataclass(frozen=True, slots=True)
class NearNode:
    a: re.Pattern[str]
    b: re.Pattern[str]
    window: int

    def find(self, text: str) -> tuple[int, int] | None:
        a_spans = [m.span() for m in self.a.finditer(text)]
        if not a_spans:
            return None
        b_spans = [m.span() for m in self.b.finditer(text)]
        if not b_spans:
            return None
        best: tuple[int, int] | None = None
        for a0, a1 in a_spans:
            for b0, b1 in b_spans:
                gap = b0 - a1 if b0 >= a1 else a0 - b1
                if gap <= self.window:
                    span = (min(a0, b0), max(a1, b1))
                    if best is None or (span[1] - span[0]) < (best[1] - best[0]):
                        best = span
        return best


def _compile_pattern(raw: object, rule_id: str) -> re.Pattern[str]:
    if not isinstance(raw, str):
        raise RuleSyntaxError(f"{rule_id}: pattern must be a string, got {type(raw).__name__}")
    if len(raw) > MAX_PATTERN_LEN:
        raise RuleSyntaxError(f"{rule_id}: pattern exceeds {MAX_PATTERN_LEN} characters")
    if _REDOS.search(raw):
        raise RuleSyntaxError(
            f"{rule_id}: pattern has nested quantifiers and risks catastrophic "
            f"backtracking in the request path: {raw!r}"
        )
    try:
        return re.compile(raw)
    except re.error as exc:
        raise RuleSyntaxError(f"{rule_id}: uncompilable regex {raw!r}: {exc}") from exc


_OPERATORS = frozenset({"regex", "all", "any", "not", "near"})


def compile_node(spec: Any, rule_id: str) -> MatchNode:
    """Compile one match-expression node.

    Requiring exactly one operator per node keeps the YAML unambiguous. A node
    with two keys reads as though both apply, but the evaluation order would be
    invented by the parser rather than stated by the author.
    """
    if not isinstance(spec, dict):
        raise RuleSyntaxError(f"{rule_id}: match node must be a mapping, got {type(spec).__name__}")

    present = _OPERATORS.intersection(spec)
    if len(present) != 1:
        raise RuleSyntaxError(
            f"{rule_id}: match node must have exactly one operator "
            f"from {sorted(_OPERATORS)}, found {sorted(present) or 'none'}"
        )

    op = next(iter(present))
    body = spec[op]

    if op == "regex":
        return RegexNode(_compile_pattern(body, rule_id))

    if op in ("all", "any"):
        if not isinstance(body, list) or not body:
            raise RuleSyntaxError(f"{rule_id}: '{op}' requires a non-empty list")
        children = tuple(compile_node(c, rule_id) for c in body)
        return AllNode(children) if op == "all" else AnyNode(children)

    if op == "not":
        return NotNode(compile_node(body, rule_id))

    if not isinstance(body, dict) or not {"a", "b"}.issubset(body):
        raise RuleSyntaxError(f"{rule_id}: 'near' requires keys 'a' and 'b'")
    window = body.get("window", 40)
    if not isinstance(window, int) or window <= 0:
        raise RuleSyntaxError(f"{rule_id}: 'near.window' must be a positive integer")
    return NearNode(
        _compile_pattern(body["a"], rule_id),
        _compile_pattern(body["b"], rule_id),
        window,
    )
