"""Combines canary, secret and exfiltration scanning into one pass over a
complete response body, and produces the redacted text a caller shows the
client (ADR-0009). `stream_scanner.py` is the streaming equivalent, built
on this same `scan()`.
"""

from __future__ import annotations

from .canary import scan_for_canary
from .exfil import scan_for_fabricated_tool_calls, scan_for_markdown_image_exfil
from .secrets import scan_for_secrets
from .types import EgressFinding, EgressScanResult, FindingKind


def scan(text: str, *, canary: str | None = None) -> EgressScanResult:
    findings: list[EgressFinding] = []

    canary_finding = scan_for_canary(text, canary) if canary else None
    if canary_finding is not None:
        findings.append(canary_finding)

    findings.extend(scan_for_secrets(text))
    findings.extend(scan_for_markdown_image_exfil(text))
    findings.extend(scan_for_fabricated_tool_calls(text))
    findings.sort(key=lambda f: f.span[0])

    canary_leaked = canary_finding is not None
    return EgressScanResult(
        findings=tuple(findings),
        # Meaningless when the canary leaked - the caller withholds the
        # whole response in that case (Decision 3) rather than showing a
        # redacted version of a confirmed-compromised reply.
        redacted_text=redact(text, findings),
        canary_leaked=canary_leaked,
    )


def redact(text: str, findings: list[EgressFinding] | tuple[EgressFinding, ...]) -> str:
    non_canary = [f for f in findings if f.kind is not FindingKind.CANARY]
    kept = _drop_overlapping(sorted(non_canary, key=lambda f: f.span[0]))

    result = text
    for finding in sorted(kept, key=lambda f: f.span[0], reverse=True):
        start, end = finding.span
        result = result[:start] + f"[REDACTED:{finding.rule}]" + result[end:]
    return result


def _drop_overlapping(findings_by_start: list[EgressFinding]) -> list[EgressFinding]:
    """Coarse heuristics (exfil.py, secrets.py) aren't guaranteed disjoint -
    keep the first (leftmost) of any overlapping pair rather than let a
    second replacement corrupt offsets already consumed by the first."""
    kept: list[EgressFinding] = []
    furthest_end = -1
    for finding in findings_by_start:
        if finding.span[0] >= furthest_end:
            kept.append(finding)
            furthest_end = finding.span[1]
    return kept
