"""Data types for L0 normalisation.

The central design commitment here is the **offset map**. Every view of the text
carries, per character, the index in the *original* input it derives from. L1
matches rules against normalised or decoded text, but a decision must be
explained in terms of what the user actually sent (NFR-5). Without offset maps
that translation is impossible, and explainability degrades to "something
matched somewhere".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ViewKind(Enum):
    """Why a view exists, and how much a rule engine should trust it."""

    CANONICAL = "canonical"
    """Lossless-by-intent normalisation: NFKC, zero-width, bidi, confusables.

    Safe to treat as the user's text. A match here carries full weight.
    """

    FOLDED = "folded"
    """Aggressive, lossy folding: leetspeak, spaced letters, combining marks.

    Higher recall, higher false-positive risk. A match here should be weighted
    down and named in the rationale, never treated as equivalent to canonical.
    """

    DECODED = "decoded"
    """Plaintext recovered from an encoded blob (base64, hex, ROT-N, ...).

    Structurally untrusted: the user embedded it deliberately obscured.
    """


@dataclass(frozen=True, slots=True)
class Transform:
    """One normalisation step that actually fired.

    `span` refers to the **original** text, so a report can highlight exactly
    what was rewritten.
    """

    name: str
    detail: str
    span: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class TextView:
    """A rendering of the input, plus the map back to where it came from."""

    kind: ViewKind
    text: str
    offset_map: tuple[int, ...]
    depth: int = 0
    provenance: str = ""

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Translate a span in this view to a span in the original text.

        Returns a half-open interval. For decoded views every character of the
        payload maps to the encoded blob, so any span inside it collapses to
        the blob's own extent - which is the honest answer: the match exists
        because of that blob, and pointing at a character offset inside
        recovered plaintext would be meaningless to the reader.
        """
        if not self.offset_map:
            return (0, 0)
        start = max(0, min(start, len(self.offset_map) - 1))
        end = max(start + 1, min(end, len(self.offset_map)))
        origins = self.offset_map[start:end]
        return (min(origins), max(origins) + 1)


@dataclass(frozen=True, slots=True)
class NormalizationReport:
    """Everything L0 learned, including that it had to work hard.

    Heavy obfuscation is signal in its own right: `obfuscation_score` and
    `max_decode_depth` are features for fusion, not merely bookkeeping.
    Legitimate prompts rarely arrive triple-base64-encoded.
    """

    original: str
    views: tuple[TextView, ...]
    transforms: tuple[Transform, ...] = ()
    max_decode_depth: int = 0
    decode_capped: bool = False
    truncated: bool = False
    obfuscation_score: float = 0.0
    elapsed_ms: float = field(default=0.0, compare=False)

    @property
    def canonical(self) -> TextView:
        for v in self.views:
            if v.kind is ViewKind.CANONICAL:
                return v
        raise AssertionError("canonical view is always constructed")

    def view(self, kind: ViewKind) -> TextView | None:
        """First view of `kind`, or None if that kind was not produced.

        FOLDED is absent when aggressive folding changed nothing - which is the
        common case for ordinary prose, and worth knowing.
        """
        for v in self.views:
            if v.kind is kind:
                return v
        return None

    def views_of(self, kind: ViewKind) -> tuple[TextView, ...]:
        return tuple(v for v in self.views if v.kind is kind)
