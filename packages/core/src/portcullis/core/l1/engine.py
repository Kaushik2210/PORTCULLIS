"""Rule loading and evaluation.

The engine consumes an L0 `NormalizationReport` rather than raw text. That is
what gives every rule de-obfuscation and decoded-payload coverage for free, and
it is why spans can be reported in the user's own coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..l0 import NormalizationReport, ViewKind
from .dsl import RuleSyntaxError, compile_node
from .types import (
    VIEW_CONFIDENCE,
    L1Result,
    Rule,
    RuleMatch,
    RuleTests,
    Scope,
    Severity,
)

SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RuleSet:
    rules: tuple[Rule, ...]
    version: int = SUPPORTED_SCHEMA_VERSION

    def subset(self, ids: set[str]) -> RuleSet:
        """Narrow to specific rules - used to test a rule in isolation."""
        return RuleSet(tuple(r for r in self.rules if r.id in ids), self.version)


def _require(mapping: dict[str, Any], key: str, rule_id: str) -> Any:
    if key not in mapping:
        raise RuleSyntaxError(f"{rule_id}: missing required field '{key}'")
    return mapping[key]


def _parse_rule(raw: Any, *, validate_tests: bool) -> Rule:
    if not isinstance(raw, dict):
        raise RuleSyntaxError(f"rule must be a mapping, got {type(raw).__name__}")

    rule_id = raw.get("id", "<no id>")
    if not isinstance(rule_id, str) or not rule_id:
        raise RuleSyntaxError("rule is missing a string 'id'")

    try:
        severity = Severity[str(_require(raw, "severity", rule_id)).upper()]
    except KeyError as exc:
        raise RuleSyntaxError(
            f"{rule_id}: unknown severity {raw.get('severity')!r}; "
            f"expected one of {[s.name.lower() for s in Severity]}"
        ) from exc

    try:
        scopes = tuple(Scope(s) for s in _require(raw, "scope", rule_id))
    except ValueError as exc:
        raise RuleSyntaxError(f"{rule_id}: unknown scope: {exc}") from exc
    if not scopes:
        raise RuleSyntaxError(f"{rule_id}: 'scope' must list at least one scope")

    try:
        views = tuple(ViewKind(v) for v in raw.get("views", ["canonical"]))
    except ValueError as exc:
        raise RuleSyntaxError(f"{rule_id}: unknown view: {exc}") from exc

    weight = raw.get("weight", 0.5)
    if not isinstance(weight, int | float) or not 0.0 < float(weight) <= 1.0:
        raise RuleSyntaxError(f"{rule_id}: 'weight' must be in (0, 1], got {weight!r}")

    tests_raw = raw.get("tests") or {}
    positive = tuple(tests_raw.get("positive") or ())
    negative = tuple(tests_raw.get("negative") or ())
    if validate_tests:
        if not positive:
            raise RuleSyntaxError(f"{rule_id}: needs at least one 'tests.positive' case")
        if not negative:
            raise RuleSyntaxError(
                f"{rule_id}: needs at least one 'tests.negative' case - a rule that "
                f"cannot say what it must not match cannot be reviewed"
            )

    return Rule(
        id=rule_id,
        name=str(_require(raw, "name", rule_id)),
        description=str(_require(raw, "description", rule_id)),
        severity=severity,
        labels=tuple(raw.get("labels") or ()),
        scope=scopes,
        views=views,
        weight=float(weight),
        matcher=compile_node(_require(raw, "match", rule_id), rule_id),
        tests=RuleTests(positive, negative),
        references=tuple(raw.get("references") or ()),
    )


def load_rules_from_string(text: str, *, validate_tests: bool = True) -> RuleSet:
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise RuleSyntaxError("ruleset must be a mapping with 'version' and 'rules'")

    version = doc.get("version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise RuleSyntaxError(
            f"unsupported ruleset version {version!r}; expected {SUPPORTED_SCHEMA_VERSION}"
        )

    raw_rules = doc.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuleSyntaxError("ruleset has no rules")

    rules = tuple(_parse_rule(r, validate_tests=validate_tests) for r in raw_rules)

    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise RuleSyntaxError(f"duplicate rule id: {rule.id}")
        seen.add(rule.id)

    return RuleSet(rules, SUPPORTED_SCHEMA_VERSION)


def load_rules_from_dir(directory: Path, *, validate_tests: bool = True) -> RuleSet:
    """Load and merge every .yaml file in `directory`.

    Files are read in sorted order so a duplicate id always reports the same
    way regardless of filesystem ordering.
    """
    merged: list[Rule] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        rs = load_rules_from_string(path.read_text(encoding="utf-8"), validate_tests=validate_tests)
        for rule in rs.rules:
            if rule.id in seen:
                raise RuleSyntaxError(f"duplicate rule id across files: {rule.id} in {path.name}")
            seen.add(rule.id)
            merged.append(rule)
    if not merged:
        raise RuleSyntaxError(f"no rules found in {directory}")
    return RuleSet(tuple(merged), SUPPORTED_SCHEMA_VERSION)


class RuleEngine:
    """Evaluates a ruleset against a normalised request."""

    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset
        # Bucket by scope once, so a scan touches only the applicable rules
        # rather than filtering the whole set per request.
        self._by_scope: dict[Scope, tuple[Rule, ...]] = {
            scope: tuple(r for r in ruleset.rules if scope in r.scope) for scope in Scope
        }
        # A literal prefilter was tried here and removed. Extracting required
        # literals from a pattern by text inspection is not sound: `\[INST\]`
        # is an escaped literal bracket, not a character class, so stripping
        # character classes discarded a required literal and the rule was
        # skipped on input it should have matched. A false negative introduced
        # by an optimisation is strictly worse than the milliseconds it saves.
        #
        # The sound version of this is author-declared literals - an explicit
        # `requires:` field validated against each rule's own positive cases -
        # which is recorded as future work in ADR-0002 rather than guessed at
        # here.

    def scan(self, report: NormalizationReport, scope: Scope) -> tuple[RuleMatch, ...]:
        """Return every rule match, with spans in the original text."""
        matches: list[RuleMatch] = []

        for rule in self._by_scope[scope]:
            for view in report.views:
                if view.kind not in rule.views:
                    continue

                span = rule.matcher.find(view.text)
                if span is None:
                    continue

                confidence = VIEW_CONFIDENCE.get(view.kind, 0.5)
                original_span = view.to_original_span(span[0], span[1])
                matched = report.original[original_span[0] : original_span[1]]

                where = "" if view.kind is ViewKind.CANONICAL else f" (in {view.kind.value} view"
                if where and view.provenance:
                    where += f": {view.provenance}"
                if where:
                    where += ")"

                matches.append(
                    RuleMatch(
                        rule_id=rule.id,
                        name=rule.name,
                        severity=rule.severity,
                        labels=rule.labels,
                        weight=round(rule.weight * confidence, 4),
                        view=view.kind,
                        span=original_span,
                        matched_text=matched[:200],
                        rationale=f"{rule.id} ({rule.name}): {rule.description}{where}",
                    )
                )
                break  # one match per rule; the first view that fires is enough

        matches.sort(key=lambda m: (-m.weight, m.rule_id))
        return tuple(matches)

    def evaluate(self, report: NormalizationReport, scope: Scope) -> L1Result:
        """Scan and reduce to a single bounded layer score.

        The aggregation is a noisy-OR: independent weak signals accumulate, but
        the result saturates rather than summing past 1. This is a *feature*
        for fusion (ADR-0001), not a verdict - fusion learns what it is worth
        relative to the other layers, so it is not tuned here.
        """
        matches = self.scan(report, scope)
        residual = 1.0
        for m in matches:
            residual *= 1.0 - min(max(m.weight, 0.0), 0.999)
        return L1Result(matches=matches, score=round(1.0 - residual, 6))


def all_rule_ids(rules: Iterable[Rule]) -> tuple[str, ...]:
    return tuple(r.id for r in rules)
