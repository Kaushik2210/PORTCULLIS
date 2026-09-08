"""Secret/PII egress scanning: named pattern rules for known shapes, plus a
Shannon-entropy check over long opaque tokens for shapes no named rule
covers (the spec's "entropy-based detection for unknown formats"). Unlike
the canary (Decision 1), every rule here carries real false-positive risk,
so a match is redacted, not grounds to block the whole response
(ADR-0009, Decision 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .entropy import shannon_entropy
from .types import EgressFinding, FindingKind, mask_preview


@dataclass(frozen=True, slots=True)
class _NamedRule:
    name: str
    pattern: re.Pattern[str]
    description: str


_NAMED_RULES: tuple[_NamedRule, ...] = (
    _NamedRule(
        "aws-access-key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "matches the AWS access key ID shape",
    ),
    _NamedRule(
        "generic-api-key",
        re.compile(r"\b(?:sk|pk|api)-[A-Za-z0-9]{20,}\b"),
        "matches a common API-key prefix convention (sk-/pk-/api-)",
    ),
    _NamedRule(
        "private-key-header",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "matches a PEM private key header",
    ),
    _NamedRule(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "matches the three-segment base64url JWT shape",
    ),
    _NamedRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "matches an email address",
    ),
    _NamedRule(
        "phone",
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "matches a phone-number-shaped digit sequence",
    ),
)

_ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{20,}")
_ENTROPY_THRESHOLD_BITS_PER_CHAR = 4.0
"""Reasoned default, not tuned against a labelled corpus (ADR-0009): random
base64/hex-alphabet data typically lands well above this; ordinary English
text and identifiers typically land well below it. The same category of
disclosed-but-unmeasured threshold as L4's risk bars (ADR-0008)."""


def scan_for_secrets(text: str) -> tuple[EgressFinding, ...]:
    findings: list[EgressFinding] = []
    covered: list[tuple[int, int]] = []

    for rule in _NAMED_RULES:
        for m in rule.pattern.finditer(text):
            findings.append(
                EgressFinding(
                    kind=FindingKind.SECRET,
                    rule=rule.name,
                    span=m.span(),
                    preview=mask_preview(m.group()),
                    rationale=f"{rule.name}: {rule.description}.",
                )
            )
            covered.append(m.span())

    for m in _ENTROPY_TOKEN.finditer(text):
        span = m.span()
        if any(span[0] < c_end and span[1] > c_start for c_start, c_end in covered):
            continue  # already explained by a named rule; don't double-report
        entropy = shannon_entropy(m.group())
        if entropy >= _ENTROPY_THRESHOLD_BITS_PER_CHAR:
            findings.append(
                EgressFinding(
                    kind=FindingKind.SECRET,
                    rule="high-entropy-token",
                    span=span,
                    preview=mask_preview(m.group()),
                    rationale=(
                        f"high-entropy-token: {len(m.group())}-character token at "
                        f"{entropy:.2f} bits/char - shaped like a secret with no matching "
                        "named rule."
                    ),
                )
            )

    findings.sort(key=lambda f: f.span[0])
    return tuple(findings)
