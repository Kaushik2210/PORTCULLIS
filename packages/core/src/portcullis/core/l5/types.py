"""Types for L5 egress inspection (ADR-0009).

A `matched_text` preview on a finding is itself something the decision
trace / logs will carry - for a real secret, storing the raw match there
would turn the audit trail into a second leak vector for the same secret.
`mask_preview()` exists specifically so nothing downstream of a finding
ever has to handle an unmasked secret; canary findings are the one
exception (the canary is a PORTCULLIS-generated tracking value, not a real
secret, so seeing it in full is the point - it's the proof)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FindingKind(Enum):
    CANARY = "canary"
    SECRET = "secret"  # noqa: S105 - an enum member name, not a credential
    EXFIL = "exfil"


def mask_preview(text: str, *, keep: int = 4) -> str:
    """`"AKIAABCDEFGHIJKLMNOP"` -> `"AKIA...MNOP"` - enough to confirm a
    real finding without the preview itself becoming a leak."""
    if len(text) <= keep * 2:
        return "*" * len(text)
    return f"{text[:keep]}...{text[-keep:]}"


@dataclass(frozen=True, slots=True)
class EgressFinding:
    kind: FindingKind
    rule: str
    span: tuple[int, int]
    preview: str
    rationale: str


@dataclass(frozen=True, slots=True)
class EgressScanResult:
    findings: tuple[EgressFinding, ...]
    redacted_text: str
    canary_leaked: bool

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)
