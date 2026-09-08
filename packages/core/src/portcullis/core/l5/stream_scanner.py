"""Streaming egress scanning: a sliding window over incrementally-arriving
text, so a canary or secret split across SSE chunk boundaries is still
caught rather than missed by scanning each chunk in isolation (ADR-0009,
Decision 4).

Rescans the *entire* accumulated buffer on every `feed()` call rather than
maintaining incremental match state - O(n) per call, O(n^2) over a full
stream. Acceptable for this milestone's scope (a demo-length response);
not claimed to scale to very long streams without revisiting, the same
honesty this ADR already applies to the TTFT penalty itself.

**A real, disclosed limitation, not silently accepted:** the window only
protects a match up to `window_chars` long - a match's own characters get
released before the pattern is recognised once the match is longer than
the window (its start ages past the release boundary before its end has
even arrived to complete the pattern). The canary's length is fully known
(this module generates it), so `__init__` refuses a `window_chars` smaller
than the canary rather than risk that silently. Secret/exfil patterns have
no such guarantee - a JWT or a multi-line private key block can exceed
`DEFAULT_WINDOW_CHARS`, and nothing here catches a match longer than the
window it's compared against. A larger window trades more TTFT for a
larger guaranteed match length; this trade-off is not tuned or benchmarked
here (Decision 4).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .scanner import redact, scan
from .types import EgressFinding, FindingKind

DEFAULT_WINDOW_CHARS = 200
"""Reasoned default (ADR-0009): comfortably larger than a canary
(19-character prefix + 32 hex characters = 51) or any named secret shape
this module recognises, so a match straddling the release boundary is
vanishingly unlikely - not a value tuned against measured chunk sizes from
a real provider."""


@dataclass(frozen=True, slots=True)
class StreamScanResult:
    release_text: str
    """Redacted text safe to forward to the client now - empty when
    nothing new has aged past the window yet, or when halted."""
    halted: bool
    """True once a canary has been confirmed - the caller must stop
    forwarding further chunks; already-sent text can't be recalled."""
    findings: tuple[EgressFinding, ...]
    """Findings newly surfaced by this call (already reflected in
    `release_text` via redaction, except a canary finding, which never
    appears in `release_text` at all)."""


def _reanchor(findings: list[EgressFinding], offset: int) -> list[EgressFinding]:
    return [replace(f, span=(f.span[0] - offset, f.span[1] - offset)) for f in findings]


class SlidingWindowScanner:
    def __init__(
        self, *, canary: str | None = None, window_chars: int = DEFAULT_WINDOW_CHARS
    ) -> None:
        """Raises if `window_chars` is smaller than `canary` - the one
        pattern length this module fully controls (it generated the
        canary), so a misconfiguration that would silently leak it
        character-by-character before the match completes is refused
        outright rather than shipped as a quiet gap. No such guarantee is
        possible for secrets/exfil patterns of unbounded length (a long
        JWT, a multi-line private key) - that limitation is real and is
        documented, not guarded against, at the module level."""
        if canary is not None and window_chars < len(canary):
            raise ValueError(
                f"window_chars ({window_chars}) is smaller than the canary "
                f"({len(canary)} chars) - characters could be released before a "
                "complete match is recognised, defeating the whole point of the window."
            )
        self._canary = canary
        self._window_chars = window_chars
        self._buffer = ""
        self._released_up_to = 0
        self._halted = False

    def feed(self, chunk: str) -> StreamScanResult:
        if self._halted:
            return StreamScanResult(release_text="", halted=True, findings=())

        self._buffer += chunk
        result = scan(self._buffer, canary=self._canary)
        if result.canary_leaked:
            self._halted = True
            return StreamScanResult(release_text="", halted=True, findings=result.findings)

        release_boundary = max(self._released_up_to, len(self._buffer) - self._window_chars)
        return self._release_up_to(release_boundary, result.findings)

    def finish(self) -> StreamScanResult:
        """Call once the upstream is exhausted, to flush whatever is still
        held back in the window."""
        if self._halted:
            return StreamScanResult(release_text="", halted=True, findings=())

        result = scan(self._buffer, canary=self._canary)
        if result.canary_leaked:
            self._halted = True
            return StreamScanResult(release_text="", halted=True, findings=result.findings)

        return self._release_up_to(len(self._buffer), result.findings)

    def _release_up_to(
        self, boundary: int, findings: tuple[EgressFinding, ...]
    ) -> StreamScanResult:
        if boundary <= self._released_up_to:
            return StreamScanResult(release_text="", halted=False, findings=())

        releasable = [
            f for f in findings if f.kind is not FindingKind.CANARY and f.span[1] <= boundary
        ]
        prefix = self._buffer[self._released_up_to : boundary]
        release_text = redact(prefix, _reanchor(releasable, self._released_up_to))
        self._released_up_to = boundary
        return StreamScanResult(release_text=release_text, halted=False, findings=tuple(releasable))
