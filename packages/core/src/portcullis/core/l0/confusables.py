"""Homoglyph folding table.

Why this is hand-curated rather than a dependency or the full Unicode file:

* **NFKC does not do this.** NFKC folds fullwidth and compatibility forms, but
  Cyrillic small a (U+0430) and Latin a (U+0061) are distinct characters with
  distinct semantics, and NFKC leaves them alone - correctly. For rendering.
  For a signature engine it means one keystroke bypasses every rule.
* **The full confusables.txt is ~6000 entries** covering every script pair. We
  only ever fold *to* ASCII, because that is what the rules are written in, so
  the vast majority of it is dead weight inside a 5ms budget.
* **A curated table is auditable.** A reviewer can read this file and see
  exactly what gets rewritten. That matters more here than coverage: a wrong
  entry silently corrupts user text.

Coverage is deliberately partial and skewed to the scripts actually used in
observed homoglyph attacks (Cyrillic, Greek). Gaps are a known limitation, and
extending the table is cheap. See docs/threat-model.md section 7.
"""

from __future__ import annotations

# Cyrillic characters that render as Latin ASCII in common fonts.
_CYRILLIC: dict[str, str] = {
    "а": "a",
    "А": "A",  # а А
    "е": "e",
    "Е": "E",  # е Е
    "о": "o",
    "О": "O",  # о О
    "р": "p",
    "Р": "P",  # р Р
    "с": "c",
    "С": "C",  # с С
    "у": "y",
    "У": "Y",  # у У
    "х": "x",
    "Х": "X",  # х Х
    "і": "i",
    "І": "I",  # і І
    "ј": "j",
    "Ј": "J",  # ј Ј
    "ѕ": "s",
    "Ѕ": "S",  # ѕ Ѕ
    "һ": "h",
    "Н": "H",  # һ Н
    "ԁ": "d",
    "Ԓ": "R",  # ԁ Ԓ
    "ԛ": "q",
    "ԝ": "w",  # ԛ ԝ
    "В": "B",
    "К": "K",  # В К
    "М": "M",
    "Т": "T",  # М Т
    "г": "r",
    "б": "b",  # г б
    "ӏ": "l",  # ӏ
}

# Greek characters that render as Latin ASCII.
_GREEK: dict[str, str] = {
    "α": "a",
    "Α": "A",  # α Α
    "ε": "e",
    "Ε": "E",  # ε Ε
    "ο": "o",
    "Ο": "O",  # ο Ο
    "ρ": "p",
    "Ρ": "P",  # ρ Ρ
    "ι": "i",
    "Ι": "I",  # ι Ι
    "κ": "k",
    "Κ": "K",  # κ Κ
    "ν": "v",
    "Ν": "N",  # ν Ν
    "τ": "t",
    "Τ": "T",  # τ Τ
    "υ": "u",
    "Υ": "Y",  # υ Υ
    "χ": "x",
    "Χ": "X",  # χ Χ
    "β": "B",
    "Β": "B",  # β Β
    "Ζ": "Z",
    "Η": "H",  # Ζ Η
    "Μ": "M",
    "σ": "o",  # Μ σ
    "ϲ": "c",
    "Ϲ": "C",  # ϲ Ϲ
}

# Latin-script and symbol lookalikes that NFKC leaves in place.
_LATIN_LOOKALIKE: dict[str, str] = {
    "ı": "i",  # ı dotless i
    "ȷ": "j",  # ȷ dotless j
    "ℓ": "l",  # ℓ script small l
    "ǀ": "l",  # ǀ latin letter dental click
    "ⱪ": "k",  # ⱪ
    "ɡ": "g",  # ɡ script g
    "ɩ": "i",  # ɩ
    "ɪ": "i",  # ɪ
    "ʀ": "R",  # ʀ
    "ʋ": "v",  # ʋ
    "ᴠ": "V",  # ᴠ
    "ᴡ": "W",  # ᴡ
    "ո": "n",  # ն armenian
    "օ": "o",  # օ armenian
    "Ꭰ": "D",  # Ꭰ cherokee
    "Ꮮ": "L",  # Ꮞ cherokee
}

CONFUSABLES: dict[str, str] = {**_CYRILLIC, **_GREEK, **_LATIN_LOOKALIKE}

# str.translate is a C-level loop; building the table once at import keeps the
# hot path free of per-character dict lookups in Python.
CONFUSABLE_TABLE: dict[int, str] = {ord(k): v for k, v in CONFUSABLES.items()}

# Invisible characters removed outright.
#
# Note ZWNJ/ZWJ (U+200C/U+200D) are semantically meaningful in Arabic, Indic
# scripts and emoji sequences. Stripping them is correct for a detector - we
# are not rendering text - but it is a real behaviour difference, not a no-op.
ZERO_WIDTH: frozenset[str] = frozenset(
    {
        "​",  # zero width space
        "‌",  # zero width non-joiner
        "‍",  # zero width joiner
        "﻿",  # zero width no-break space / BOM
        "⁠",  # word joiner
        "­",  # soft hyphen
        "᠎",  # mongolian vowel separator
        "͏",  # combining grapheme joiner
    }
)

# Bidirectional control characters. These reorder rendered text without
# changing its logical content, so what a reviewer reads and what a model
# receives can differ arbitrarily.
BIDI_CONTROLS: frozenset[str] = frozenset(
    {
        "‪",
        "‫",
        "‬",
        "‭",
        "‮",
        "⁦",
        "⁧",
        "⁨",
        "⁩",
        "‎",
        "‏",
        "؜",
    }
)

# Leetspeak substitutions. Applied only in the FOLDED view: '0' -> 'o' is
# obviously wrong for "port 8080" and would wreck hashes, IDs and passwords.
LEET: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "+": "t",
    "|": "l",
    "(": "c",
}
LEET_TABLE: dict[int, str] = {ord(k): v for k, v in LEET.items()}
