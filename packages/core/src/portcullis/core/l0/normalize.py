"""The L0 pipeline.

Order is load-bearing: normalise before decoding, because an attacker can wrap
a payload in fullwidth or zero-width characters specifically to stop the base64
regex from matching it.

Everything runs inside the Tier-A budget (ADR-0002: L0+L1 p95 < 5ms at 2KB), so
the hot path is built around a fast rejection test - clean printable ASCII, the
overwhelmingly common case, exits after a single scan with no allocation.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from .confusables import (
    BIDI_CONTROLS,
    CONFUSABLE_TABLE,
    LEET_TABLE,
    ZERO_WIDTH,
)
from .decode import MAX_DECODE_DEPTH, decode_layered
from .types import NormalizationReport, TextView, Transform, ViewKind

MAX_INPUT_CHARS = 200_000

# A run of this many single-character tokens is folded to a word. Four is the
# threshold at which English stops producing them naturally: "I a m a" does not
# occur, but "I am a cat" contains runs of length 1.
SPACED_RUN_MIN = 4

_SPACED = re.compile(rf"(?:(?<=\s)|\A)(?:[A-Za-z]\s+){{{SPACED_RUN_MIN - 1},}}[A-Za-z](?=\s|\Z)")


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    max_chars: int = MAX_INPUT_CHARS
    max_decode_depth: int = MAX_DECODE_DEPTH
    fold_leet: bool = True
    fold_spaced: bool = True
    decode: bool = True


DEFAULT_CONFIG = NormalizerConfig()


def _is_fast_path(text: str) -> bool:
    """True when no canonical transform can possibly apply.

    Printable ASCII contains no confusables, no zero-width characters, no bidi
    controls, and is a fixed point of NFKC. Checking that up front keeps
    ordinary traffic - which is nearly all traffic - off the slow path.
    """
    return text.isascii() and text.isprintable()


def _canonicalise(text: str) -> tuple[str, list[int], list[Transform]]:
    """Apply canonical transforms character by character, tracking origins.

    Per-character NFKC rather than whole-string: it keeps the offset map exact.
    Whole-string NFKC composes across character boundaries, which would make a
    single output character derive from several inputs and force the map to
    lie. The compatibility foldings that matter here (fullwidth, ligatures,
    circled digits) are all single-character, so nothing is lost.
    """
    out: list[str] = []
    origins: list[int] = []
    transforms: list[Transform] = []

    for i, ch in enumerate(text):
        if ch in ZERO_WIDTH:
            transforms.append(Transform("zero_width", f"stripped U+{ord(ch):04X}", (i, i + 1)))
            continue
        if ch in BIDI_CONTROLS:
            transforms.append(
                Transform("bidi", f"stripped bidi control U+{ord(ch):04X}", (i, i + 1))
            )
            continue

        nfkc = unicodedata.normalize("NFKC", ch)
        if nfkc != ch:
            transforms.append(Transform("nfkc", f"U+{ord(ch):04X} -> {nfkc!r}", (i, i + 1)))

        # Confusables are folded on the NFKC *output*, not the input. A
        # compatibility mapping can land on a confusable - mathematical bold
        # small alpha decomposes to Greek alpha, which only then folds to 'a'.
        # Folding first would miss that and break idempotence, so that a second
        # pass over the canonical form produced a different result. Hypothesis
        # found this; the example-based tests did not.
        for c in nfkc:
            if c in ZERO_WIDTH or c in BIDI_CONTROLS:
                continue
            folded = CONFUSABLE_TABLE.get(ord(c))
            if folded is None:
                out.append(c)
                origins.append(i)
                continue
            transforms.append(
                Transform("confusable", f"U+{ord(c):04X} {c!r} -> {folded!r}", (i, i + 1))
            )
            for f in folded:
                out.append(f)
                origins.append(i)

    return "".join(out), origins, transforms


def _fold_aggressive(text: str, origins: list[int]) -> tuple[str, list[int]] | None:
    """Leetspeak and spaced-letter folding. Lossy by design.

    Returns None when nothing changed, so the FOLDED view is absent rather than
    a duplicate of canonical - "no aggressive folding applied" is information.
    """
    offs = list(origins)

    # str.translate is a C-level loop and all leet substitutions are 1:1, so
    # this cannot change length and the offset map stays valid untouched.
    folded = text.translate(LEET_TABLE)
    changed = folded != text

    # Collapse runs of spaced single letters: "i g n o r e" -> "ignore".
    collapsed_parts: list[str] = []
    collapsed_offs: list[int] = []
    last = 0
    for m in _SPACED.finditer(folded):
        collapsed_parts.append(folded[last : m.start()])
        collapsed_offs.extend(offs[last : m.start()])
        for j, ch in enumerate(folded[m.start() : m.end()]):
            if not ch.isspace():
                collapsed_parts.append(ch)
                collapsed_offs.append(offs[m.start() + j])
        changed = True
        last = m.end()
    collapsed_parts.append(folded[last:])
    collapsed_offs.extend(offs[last:])

    result = "".join(collapsed_parts)
    if not changed or result == text:
        return None
    return result, collapsed_offs


def _obfuscation_score(
    *,
    transforms: list[Transform],
    depth: int,
    capped: bool,
    length: int,
) -> float:
    """Collapse 'how hard did we have to work' into one bounded scalar.

    Weights are hand-set and deliberately coarse. This is a *feature* consumed
    by learned fusion (ADR-0001), not a verdict, so its calibration is fusion's
    problem rather than something to tune by hand here.
    """
    if length == 0:
        return 0.0

    kinds = {t.name for t in transforms}
    rewrites = sum(1 for t in transforms if t.name in {"confusable", "nfkc"})

    score = 0.0
    score += min(depth, 3) * 0.22
    score += 0.15 if capped else 0.0
    score += 0.20 if "zero_width" in kinds else 0.0
    score += 0.25 if "bidi" in kinds else 0.0
    score += min(rewrites / max(length, 1), 0.3) * 0.6
    return max(0.0, min(1.0, score))


def normalize(text: str, config: NormalizerConfig = DEFAULT_CONFIG) -> NormalizationReport:
    """Normalise, de-obfuscate, and report everything that was done.

    Never raises on any input, including lone surrogates and malformed
    sequences: L0 sees every byte of hostile traffic before any other layer,
    so a crash here is an availability bug in the whole gateway.
    """
    started = time.perf_counter()

    truncated = len(text) > config.max_chars
    original = text[: config.max_chars] if truncated else text

    if _is_fast_path(original):
        canonical = TextView(
            kind=ViewKind.CANONICAL,
            text=original,
            offset_map=tuple(range(len(original))),
        )
        transforms: list[Transform] = []
        canon_text, origins = original, list(range(len(original)))
    else:
        canon_text, origins, transforms = _canonicalise(original)
        canonical = TextView(
            kind=ViewKind.CANONICAL,
            text=canon_text,
            offset_map=tuple(origins),
        )

    views: list[TextView] = [canonical]

    if config.fold_leet or config.fold_spaced:
        folded = _fold_aggressive(canon_text, origins)
        if folded is not None:
            views.append(
                TextView(
                    kind=ViewKind.FOLDED,
                    text=folded[0],
                    offset_map=tuple(folded[1]),
                    provenance="leet+spaced",
                )
            )

    max_depth = 0
    capped = False
    if config.decode:
        decodings, capped = decode_layered(canon_text, config.max_decode_depth)
        for dec in decodings:
            # Every decoded character maps to the encoded blob it came from:
            # an offset inside recovered plaintext has no meaning in the
            # user's original text.
            lo = origins[dec.span[0]] if dec.span[0] < len(origins) else 0
            hi = origins[min(dec.span[1], len(origins)) - 1] if origins else 0
            anchor = lo if lo <= hi else hi
            views.append(
                TextView(
                    kind=ViewKind.DECODED,
                    text=dec.text,
                    offset_map=tuple([anchor] * len(dec.text)),
                    depth=dec.depth,
                    provenance=f"{dec.scheme}@[{lo}:{hi + 1}]",
                )
            )
            max_depth = max(max_depth, dec.depth)
            transforms.append(Transform(dec.scheme, f"decoded at depth {dec.depth}", (lo, hi + 1)))

    return NormalizationReport(
        original=original,
        views=tuple(views),
        transforms=tuple(transforms),
        max_decode_depth=max_depth,
        decode_capped=capped,
        truncated=truncated,
        obfuscation_score=_obfuscation_score(
            transforms=transforms, depth=max_depth, capped=capped, length=len(original)
        ),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
