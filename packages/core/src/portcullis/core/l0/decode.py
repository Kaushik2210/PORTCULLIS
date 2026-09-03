"""Layered de-obfuscation.

Decode, then hand the plaintext back for re-scanning. Three constraints shape
every decision here:

* **A decoder is an amplifier, so it is a DoS surface.** Recursion is capped,
  the number of blobs per pass is capped, and output size is capped. The caps
  are security controls, not tuning knobs.
* **A decoder is also a false-positive surface.** Any long alphanumeric run
  "is" valid base64. Decoding an API key or a UUID and scanning the resulting
  mojibake manufactures signal from nothing. Every decoder therefore has to
  prove its output looks like text before the result is accepted.
* **Failure is normal.** Most candidate blobs are not encoded payloads. Rejection
  is the expected path and must be cheap.
"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass

MAX_DECODE_DEPTH = 6
MAX_BLOBS_PER_PASS = 8
MIN_BLOB_LEN = 16
MAX_DECODED_CHARS = 8192

_B64 = re.compile(rf"[A-Za-z0-9+/]{{{MIN_BLOB_LEN},}}={{0,2}}")
_B32 = re.compile(rf"[A-Z2-7]{{{MIN_BLOB_LEN},}}={{0,6}}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
# Percent-escapes are usually scattered through a value rather than adjacent
# ("a%20b%20c"), so this counts occurrences anywhere and the whole text is then
# unquoted as one candidate - matching how URL encoding is actually applied.
_PCT_ONE = re.compile(r"%[0-9a-fA-F]{2}")
_MORSE = re.compile(r"[.\-]{1,6}(?:[ /]+[.\-]{1,6}){4,}")

# A small closed-class vocabulary. Used to decide whether a *candidate*
# decoding is English at all - it is not a detection signal and carries no
# attack-specific vocabulary, so it cannot be gamed into a false negative.
_COMMON = frozenset(
    """the be to of and a in that have it for not on with as you do at this but his
    by from they we say her she or an will my one all would there their what so up
    out if about who get which go me when make can like time no just him know take
    into your good some could them see other than then now look only come its over
    think also back after use two how our work first well way even new want because
    any these give day most us is are was were been has had please write read show
    tell make give system user prompt rule instruction instructions ignore previous
    above below reveal print output secret key password token admin role act""".split()  # noqa: SIM905 - a 100-word list literal would be far less readable
)

_MORSE_TABLE = {
    ".-": "a",
    "-...": "b",
    "-.-.": "c",
    "-..": "d",
    ".": "e",
    "..-.": "f",
    "--.": "g",
    "....": "h",
    "..": "i",
    ".---": "j",
    "-.-": "k",
    ".-..": "l",
    "--": "m",
    "-.": "n",
    "---": "o",
    ".--.": "p",
    "--.-": "q",
    ".-.": "r",
    "...": "s",
    "-": "t",
    "..-": "u",
    "...-": "v",
    ".--": "w",
    "-..-": "x",
    "-.--": "y",
    "--..": "z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}


@dataclass(frozen=True, slots=True)
class Decoded:
    """One successfully decoded blob."""

    text: str
    scheme: str
    span: tuple[int, int]
    depth: int


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for ch in s if ch.isprintable() or ch in "\n\r\t")
    return ok / len(s)


def looks_like_text(s: str, *, min_printable: float = 0.5) -> bool:
    """Reject decodings that produced binary noise.

    Deliberately permissive on the printable fraction: an attacker may append
    padding or trailing garbage to a real payload, and rejecting the whole blob
    because its tail is bytes would be an easy bypass. The English check below
    is what actually gates the ambiguous schemes.
    """
    return len(s) >= 4 and _printable_ratio(s) >= min_printable


def english_score(s: str) -> int:
    """Number of distinct common English words present.

    Used only where a decoding is otherwise unfalsifiable - every string is a
    valid ROT-N of some other string, so 'did this produce language?' is the
    only available acceptance test.
    """
    words = re.findall(r"[a-z]+", s.lower())
    return len({w for w in words if w in _COMMON})


def _try_base64(blob: str) -> str | None:
    if len(blob) % 4:
        blob = blob[: len(blob) // 4 * 4]
    if len(blob) < MIN_BLOB_LEN:
        return None
    try:
        raw = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError):
        return None
    out = raw.decode("utf-8", errors="replace")
    return out if looks_like_text(out) else None


def _try_base32(blob: str) -> str | None:
    pad = (-len(blob)) % 8
    try:
        raw = base64.b32decode(blob + "=" * pad)
    except (binascii.Error, ValueError):
        return None
    out = raw.decode("utf-8", errors="replace")
    return out if looks_like_text(out) and english_score(out) >= 1 else None


def _try_hex(blob: str) -> str | None:
    if len(blob) % 2:
        blob = blob[:-1]
    try:
        raw = bytes.fromhex(blob)
    except ValueError:
        return None
    out = raw.decode("utf-8", errors="replace")
    return out if looks_like_text(out) else None


def _try_percent(blob: str) -> str | None:
    out = urllib.parse.unquote(blob, errors="replace")
    return out if out != blob and looks_like_text(out) else None


def _build_caesar_tables() -> tuple[dict[int, str], ...]:
    """Precompute all 26 rotations as str.translate tables.

    Built once at import. A per-character Python loop over 25 candidate
    rotations was the single largest cost in L0 - roughly 50k interpreted
    operations for a 2KB input - and blew the Tier-A budget on its own.
    """
    tables: list[dict[int, str]] = []
    for n in range(26):
        t: dict[int, str] = {}
        for i in range(26):
            t[97 + i] = chr((i + n) % 26 + 97)
            t[65 + i] = chr((i + n) % 26 + 65)
        tables.append(t)
    return tuple(tables)


_CAESAR_TABLES = _build_caesar_tables()

_HAS_WORD = re.compile(r"[a-zA-Z]{3,}")


def _caesar(text: str, n: int) -> str:
    return text.translate(_CAESAR_TABLES[n % 26])


def _try_rot(text: str) -> tuple[str, str] | None:
    """Try every Caesar rotation, accept the one that yields English.

    ROT-N is unfalsifiable by construction, so acceptance is purely 'is the
    output language?'. Requiring two distinct common words keeps short strings
    from rotating into an accidental match.

    The early return matters as much as the test: text that is *already*
    English is not ROT-encoded, and skipping the 25-rotation search for it
    keeps ordinary traffic - nearly all traffic - off this path entirely.
    """
    if not _HAS_WORD.search(text):
        return None
    base = english_score(text)
    if base >= 2:
        return None

    best: tuple[int, str, int] | None = None
    for n in range(1, 26):
        rotated = _caesar(text, n)
        score = english_score(rotated)
        if score >= 2 and score > base and (best is None or score > best[2]):
            best = (n, rotated, score)
    if best is None:
        return None
    return (best[1], f"rot{best[0]}")


def _try_morse(blob: str) -> str | None:
    letters = re.split(r"[ ]*/[ ]*|\s+", blob.strip())
    out: list[str] = []
    for sym in letters:
        if not sym:
            out.append(" ")
            continue
        ch = _MORSE_TABLE.get(sym)
        if ch is None:
            return None
        out.append(ch)
    text = "".join(out)
    return text if len(text) >= 4 else None


def decode_pass(text: str, depth: int) -> list[Decoded]:
    """One decoding pass over `text`. Non-recursive; the caller drives depth."""
    found: list[Decoded] = []

    def add(match: re.Match[str], result: str | None, scheme: str) -> None:
        if result and len(found) < MAX_BLOBS_PER_PASS:
            found.append(Decoded(result[:MAX_DECODED_CHARS], scheme, match.span(), depth))

    for m in _B64.finditer(text):
        add(m, _try_base64(m.group()), "base64")
    for m in _HEX.finditer(text):
        add(m, _try_hex(m.group()), "hex")
    for m in _B32.finditer(text):
        add(m, _try_base32(m.group()), "base32")
    for m in _MORSE.finditer(text):
        add(m, _try_morse(m.group()), "morse")

    # Whole-text schemes. Neither percent-encoding nor a Caesar shift has a
    # delimiter to locate, so both are only meaningful applied to everything.
    if len(_PCT_ONE.findall(text)) >= 2 and len(found) < MAX_BLOBS_PER_PASS:
        unquoted = _try_percent(text)
        if unquoted is not None:
            found.append(Decoded(unquoted[:MAX_DECODED_CHARS], "url", (0, len(text)), depth))

    rot = _try_rot(text)
    if rot is not None and len(found) < MAX_BLOBS_PER_PASS:
        found.append(Decoded(rot[0][:MAX_DECODED_CHARS], rot[1], (0, len(text)), depth))

    return found


def decode_layered(text: str, max_depth: int = MAX_DECODE_DEPTH) -> tuple[list[Decoded], bool]:
    """Decode recursively up to `max_depth`.

    Returns (decodings, capped). `capped` is True when the recursion limit was
    the reason we stopped rather than running out of decodable content - that
    distinction is itself a signal, since legitimate text does not bottom out
    a decoder.
    """
    results: list[Decoded] = []
    frontier = [(text, 0)]
    capped = False
    seen: set[str] = {text}

    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            capped = True
            continue
        for dec in decode_pass(current, depth + 1):
            if dec.text in seen:
                continue  # a fixed point; recursing further cannot terminate
            seen.add(dec.text)
            results.append(dec)
            if len(results) >= MAX_BLOBS_PER_PASS * 2:
                return results, capped
            frontier.append((dec.text, dec.depth))

    return results, capped
