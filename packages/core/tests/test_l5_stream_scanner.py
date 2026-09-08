"""SlidingWindowScanner - the streaming case, including a canary split
across chunk boundaries (the whole reason the window exists, ADR-0009
Decision 4).
"""

from __future__ import annotations

import pytest

from portcullis.core.l5 import SlidingWindowScanner, generate_canary


def _feed_all(scanner: SlidingWindowScanner, chunks: list[str]) -> tuple[str, bool]:
    released = ""
    halted = False
    for chunk in chunks:
        result = scanner.feed(chunk)
        released += result.release_text
        halted = halted or result.halted
        if halted:
            break
    if not halted:
        final = scanner.finish()
        released += final.release_text
        halted = final.halted
    return released, halted


def test_clean_stream_is_released_in_full() -> None:
    scanner = SlidingWindowScanner()
    chunks = ["Hello ", "there, ", "how can ", "I help ", "you today?"]
    released, halted = _feed_all(scanner, chunks)
    assert released == "".join(chunks)
    assert halted is False


def test_a_canary_entirely_within_one_chunk_halts_the_stream() -> None:
    canary = generate_canary()
    scanner = SlidingWindowScanner(canary=canary)
    chunks = ["Sure, here it is: ", canary, " - hope that helps"]
    released, halted = _feed_all(scanner, chunks)
    assert halted is True
    assert canary not in released


def test_a_canary_split_across_many_small_chunks_is_still_caught() -> None:
    canary = generate_canary()
    # The default window (200) is comfortably larger than a canary
    # (~51 chars) - the window must be, or the guard in __init__ would
    # have refused this construction (see its docstring for why).
    scanner = SlidingWindowScanner(canary=canary)
    # Feed it one character at a time - the worst case for a naive
    # per-chunk-only scan, which is exactly what the sliding window exists
    # to survive. Chunk size and window size are independent: this tests
    # tiny *chunks* against a window already sized to protect the canary.
    chunks = ["leaked: ", *list(canary), " done"]
    released, halted = _feed_all(scanner, chunks)
    assert halted is True
    assert canary not in released


def test_window_smaller_than_the_canary_is_refused_at_construction() -> None:
    canary = generate_canary()
    with pytest.raises(ValueError, match="window_chars"):
        SlidingWindowScanner(canary=canary, window_chars=10)


def test_nothing_is_released_before_the_window_fills() -> None:
    scanner = SlidingWindowScanner(window_chars=200)
    result = scanner.feed("short chunk")
    assert result.release_text == ""
    assert result.halted is False


def test_finish_flushes_whatever_is_still_held_back() -> None:
    scanner = SlidingWindowScanner(window_chars=200)
    scanner.feed("short chunk")
    final = scanner.finish()
    assert final.release_text == "short chunk"


def test_a_secret_split_across_chunks_is_redacted_not_leaked() -> None:
    # window_chars must comfortably exceed the pattern's own length for the
    # same reason the canary guard exists (stream_scanner.py's docstring) -
    # unlike the canary, nothing enforces this for secrets, so the test
    # picks a window that actually protects a 17-character email.
    scanner = SlidingWindowScanner(window_chars=30)
    email = "test@example.com"
    chunks = ["contact ", *list(email), " for help"]
    released, halted = _feed_all(scanner, chunks)
    assert halted is False
    assert email not in released
    assert "[REDACTED:email]" in released


def test_feeding_after_halted_returns_empty_and_stays_halted() -> None:
    canary = generate_canary()
    scanner = SlidingWindowScanner(canary=canary)
    scanner.feed(canary)
    assert scanner.feed("more text").halted is True
    assert scanner.feed("more text").release_text == ""


def test_released_text_is_never_produced_twice() -> None:
    scanner = SlidingWindowScanner(window_chars=5)
    seen = ""
    for chunk in ["one two three four five six seven eight nine ten"]:
        for ch in chunk:
            result = scanner.feed(ch)
            seen += result.release_text
    seen += scanner.finish().release_text
    assert seen == "one two three four five six seven eight nine ten"
