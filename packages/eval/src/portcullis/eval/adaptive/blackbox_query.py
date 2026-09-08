"""Iterative black-box query attacks (ADR-0010, Decision 5): greedy
hill-climbing over candidate single-step mutations, scored by the same
fused score a real request would get. A simplified relative of established
black-box adversarial-text search (e.g. TextFooler's greedy word-importance
search), not a reimplementation of any of them - the mutation pool here is
four cheap, generic transforms, not a learned or corpus-derived one.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from portcullis.core.l0.confusables import CONFUSABLES

_ASCII_TO_CONFUSABLE: dict[str, str] = {}
for _confusable, _ascii_char in CONFUSABLES.items():
    _ASCII_TO_CONFUSABLE.setdefault(_ascii_char, _confusable)


@dataclass(frozen=True, slots=True)
class BlackBoxAttackResult:
    seed: str
    final_text: str
    initial_score: float
    final_score: float
    queries_used: int
    evaded: bool


def _mutations(text: str, rng: random.Random) -> list[str]:
    """A handful of candidate single-step mutations from `text` - deletion,
    whitespace injection, a case flip, and a confusable substitution (which
    L0's own folding is expected to undo most of the time - included as an
    honest control, not because it's expected to work)."""
    candidates: list[str] = []
    words = text.split(" ")

    if len(words) > 3:
        idx = rng.randrange(len(words))
        candidates.append(" ".join(words[:idx] + words[idx + 1 :]))

    if text:
        idx = rng.randrange(len(text))
        candidates.append(text[:idx] + " " + text[idx:])

    if text:
        idx = rng.randrange(len(text))
        ch = text[idx]
        if ch.isalpha():
            flipped = ch.upper() if ch.islower() else ch.lower()
            candidates.append(text[:idx] + flipped + text[idx + 1 :])

    for ascii_char, confusable in _ASCII_TO_CONFUSABLE.items():
        if ascii_char in text:
            candidates.append(text.replace(ascii_char, confusable, 1))
            break

    return candidates


def run_blackbox_attack(
    seed: str,
    score_fn: Callable[[str], float],
    *,
    block_threshold: float,
    max_queries: int = 100,
    rng_seed: int = 0,
) -> BlackBoxAttackResult:
    """Greedy hill-climbing: try every candidate mutation from the current
    text, keep whichever reduces the score the most, repeat until the score
    drops below `block_threshold` (evasion) or `max_queries` is exhausted
    (failure - reported honestly, per ADR-0010 Decision 5, not hidden)."""
    rng = random.Random(rng_seed)  # noqa: S311 - a reproducible search seed, not cryptography
    current = seed
    current_score = score_fn(current)
    initial_score = current_score
    queries = 1

    while queries < max_queries and current_score >= block_threshold:
        candidates = _mutations(current, rng)
        if not candidates:
            break

        best_candidate: str | None = None
        best_score = current_score
        for candidate in candidates:
            if queries >= max_queries:
                break
            score = score_fn(candidate)
            queries += 1
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break  # no available mutation improved on the current text - stuck

        current, current_score = best_candidate, best_score

    return BlackBoxAttackResult(
        seed=seed,
        final_text=current,
        initial_score=initial_score,
        final_score=current_score,
        queries_used=queries,
        evaded=current_score < block_threshold,
    )
