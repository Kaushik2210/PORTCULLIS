"""Rule-based paraphrase attacks (ADR-0010, Decision 5): a small
hand-authored synonym/phrasing table applied to real seed attack strings
from the test set. Deliberately not LLM-generated - the point is measuring
whether cheap, mechanical rephrasing evades detection, the realistic
low-effort end of what an attacker tries first, not the most sophisticated
paraphrase imaginable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ignore": ("disregard", "forget", "bypass", "skip"),
    "previous": ("prior", "earlier", "preceding"),
    "instructions": ("directions", "rules", "guidelines", "commands"),
    "reveal": ("show", "expose", "display", "disclose"),
    "system prompt": ("system message", "initial instructions", "configuration"),
    "override": ("replace", "supersede", "bypass"),
    "pretend": ("imagine", "roleplay", "act as if"),
    "you are now": ("from now on you are", "your new role is"),
    "disregard": ("ignore", "forget", "skip"),
}


@dataclass(frozen=True, slots=True)
class ParaphraseAttempt:
    original: str
    paraphrased: str
    substitution: str


def generate_paraphrases(text: str, *, max_variants: int = 10) -> list[ParaphraseAttempt]:
    """One substitution per variant (not compounded) - each variant tests
    a single, cheap rephrasing in isolation, so a result can be attributed
    to a specific substitution rather than a combination of several."""
    variants: list[ParaphraseAttempt] = []
    lower = text.lower()

    for phrase, synonyms in _SYNONYMS.items():
        if phrase not in lower:
            continue
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for synonym in synonyms:
            paraphrased = pattern.sub(synonym, text, count=1)
            if paraphrased != text:
                variants.append(
                    ParaphraseAttempt(
                        original=text, paraphrased=paraphrased, substitution=f"{phrase}->{synonym}"
                    )
                )
            if len(variants) >= max_variants:
                return variants

    return variants
