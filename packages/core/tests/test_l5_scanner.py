"""The combined scan()/redact() pass - canary + secrets + exfil in one call
over a complete response body (ADR-0009, Decisions 1-3).
"""

from __future__ import annotations

from portcullis.core.l5 import FindingKind, generate_canary, redact, scan


def test_clean_response_has_no_findings_and_is_unchanged() -> None:
    text = "Sure, I'd be happy to help with that."
    result = scan(text)
    assert result.findings == ()
    assert result.canary_leaked is False
    assert result.redacted_text == text


def test_a_leaked_canary_is_reported_as_canary_leaked() -> None:
    canary = generate_canary()
    result = scan(f"Here's the context: {canary}", canary=canary)
    assert result.canary_leaked is True
    assert any(f.kind is FindingKind.CANARY for f in result.findings)


def test_no_canary_argument_means_no_canary_checking() -> None:
    # A response that happens to contain canary-shaped text is not a leak
    # if this call was never given a canary to check against.
    result = scan("PORTCULLIS-CANARY-deadbeef is just a string here")
    assert result.canary_leaked is False


def test_secrets_are_redacted_in_the_output_text() -> None:
    result = scan("my email is test@example.com, reach out anytime")
    assert "test@example.com" not in result.redacted_text
    assert "[REDACTED:email]" in result.redacted_text


def test_redacted_text_preserves_surrounding_content() -> None:
    result = scan("before test@example.com after")
    assert result.redacted_text.startswith("before ")
    assert result.redacted_text.endswith(" after")


def test_multiple_findings_are_all_redacted() -> None:
    result = scan("email test@example.com and call 555-123-4567 please")
    assert "test@example.com" not in result.redacted_text
    assert "555-123-4567" not in result.redacted_text


def test_redact_is_usable_standalone_on_a_findings_list() -> None:
    result = scan("contact test@example.com")
    assert redact("contact test@example.com", result.findings) == result.redacted_text
