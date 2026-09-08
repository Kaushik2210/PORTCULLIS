"""Canary generation, injection and detection - the zero-FP-by-construction
signal ADR-0009 builds everything else around.
"""

from __future__ import annotations

from portcullis.core.l5 import (
    FindingKind,
    generate_canary,
    inject_into_system_prompt,
    scan_for_canary,
)


def test_generated_canaries_are_unique() -> None:
    canaries = {generate_canary() for _ in range(1000)}
    assert len(canaries) == 1000


def test_generated_canary_has_the_expected_prefix() -> None:
    assert generate_canary().startswith("PORTCULLIS-CANARY-")


def test_injection_appends_the_canary_to_the_system_prompt() -> None:
    canary = generate_canary()
    injected = inject_into_system_prompt("You are a helpful assistant.", canary)
    assert injected.startswith("You are a helpful assistant.")
    assert canary in injected


def test_scan_finds_a_leaked_canary() -> None:
    canary = generate_canary()
    response = f"Sure, here is the internal context: {canary} - hope that helps!"
    finding = scan_for_canary(response, canary)
    assert finding is not None
    assert finding.kind is FindingKind.CANARY
    assert finding.preview == canary
    response_span = response[finding.span[0] : finding.span[1]]
    assert response_span == canary


def test_scan_finds_nothing_when_the_canary_never_appears() -> None:
    canary = generate_canary()
    assert scan_for_canary("A perfectly ordinary response.", canary) is None


def test_a_different_canary_is_not_mistaken_for_a_match() -> None:
    canary_a = generate_canary()
    canary_b = generate_canary()
    assert scan_for_canary(f"leaked: {canary_b}", canary_a) is None
