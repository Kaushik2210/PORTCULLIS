"""Named secret-pattern rules and the entropy-based fallback for unknown
shapes (ADR-0009, Decision 2). Every rule needs a true-positive and a
false-positive case, the same discipline L1's rules follow (ADR-0001).
"""

from __future__ import annotations

from portcullis.core.l5 import FindingKind, scan_for_secrets


def test_no_findings_in_ordinary_text() -> None:
    assert scan_for_secrets("The weather today is sunny with a light breeze.") == ()


def test_aws_access_key_shape_is_caught() -> None:
    findings = scan_for_secrets("export AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP")
    assert any(f.rule == "aws-access-key" for f in findings)


def test_aws_access_key_preview_does_not_contain_the_full_key() -> None:
    key = "AKIAABCDEFGHIJKLMNOP"
    findings = scan_for_secrets(f"key: {key}")
    finding = next(f for f in findings if f.rule == "aws-access-key")
    assert key not in finding.preview
    assert finding.preview.startswith(key[:4])
    assert finding.preview.endswith(key[-4:])


def test_generic_api_key_prefix_is_caught() -> None:
    findings = scan_for_secrets("Bearer sk-abcdefghijklmnopqrstuvwxyz123456")
    assert any(f.rule == "generic-api-key" for f in findings)


def test_private_key_header_is_caught() -> None:
    findings = scan_for_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
    assert any(f.rule == "private-key-header" for f in findings)


def test_jwt_shape_is_caught() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dGhpc2lzbm90cmVhbA"
    findings = scan_for_secrets(f"token={jwt}")
    assert any(f.rule == "jwt" for f in findings)


def test_email_is_caught_but_not_flagged_as_high_entropy_too() -> None:
    findings = scan_for_secrets("Contact us at support@example.com for help.")
    rules = [f.rule for f in findings]
    assert rules == ["email"]  # no duplicate high-entropy finding on the same span


def test_phone_number_is_caught() -> None:
    findings = scan_for_secrets("Call me at 555-123-4567 tomorrow.")
    assert any(f.rule == "phone" for f in findings)


def test_ordinary_prose_is_not_flagged_as_high_entropy() -> None:
    text = "This is a perfectly ordinary sentence with no secrets in it whatsoever today."
    findings = scan_for_secrets(text)
    assert not any(f.rule == "high-entropy-token" for f in findings)


def test_unknown_high_entropy_token_is_caught_by_the_entropy_fallback() -> None:
    # No named rule matches this shape, but it's random-looking and long.
    token = "kX9qP2vN7mZ4wR8tY1uJ6hL3sF5dA0cE"
    findings = scan_for_secrets(f"the internal token is {token} - keep it safe")
    assert any(f.rule == "high-entropy-token" for f in findings)


def test_all_findings_are_kind_secret() -> None:
    findings = scan_for_secrets("email me at test@example.com or call 555-000-1111")
    assert all(f.kind is FindingKind.SECRET for f in findings)


def test_findings_are_sorted_by_position() -> None:
    findings = scan_for_secrets("first test@example.com then 555-123-9999 at the end")
    starts = [f.span[0] for f in findings]
    assert starts == sorted(starts)
