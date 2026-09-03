"""L0 property tests: invariants that must hold for *any* input.

L0 is the one layer that sees every byte of hostile traffic before anything
else, unconditionally. It is also the layer most likely to be handed
deliberately malformed Unicode. Example-based tests cannot cover that input
space; these state the invariants directly and let Hypothesis attack them.

The milestone requirement is that no transform ever crashes or loops. Both are
asserted here, along with the structural invariants that make offsets usable.
"""

from __future__ import annotations

import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from portcullis.core.l0 import ViewKind, normalize

# Deliberately includes surrogates, unassigned planes, control characters and
# private-use areas: an attacker is not limited to well-formed text.
ANY_TEXT = st.text(
    alphabet=st.characters(codec="utf-8"),
    max_size=2000,
)

CLEAN_ASCII = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    max_size=500,
)

PROFILE = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@PROFILE
@given(ANY_TEXT)
def test_never_raises(text: str) -> None:
    """No input may crash the normaliser. It runs before every other layer."""
    normalize(text)


@PROFILE
@given(ANY_TEXT)
def test_terminates_promptly(text: str) -> None:
    """Guards against a decode loop or catastrophic backtracking.

    L0 sits inside a 5ms p95 budget (ADR-0002). One second is not the SLO -
    it is the threshold above which something is structurally wrong.
    """
    started = time.perf_counter()
    normalize(text)
    assert time.perf_counter() - started < 1.0


@PROFILE
@given(ANY_TEXT)
def test_output_is_bounded(text: str) -> None:
    """Decoding must not be an expansion bomb.

    base64 shrinks, but hex/URL/morse decoding and repeated view generation
    could otherwise let a small input produce a large working set.
    """
    report = normalize(text)
    total = sum(len(v.text) for v in report.views)
    assert total <= 20 * max(len(text), 1) + 10_000


@PROFILE
@given(ANY_TEXT)
def test_offset_map_is_structurally_valid(text: str) -> None:
    """Every view must be able to point at the original text.

    Without this, a rule match in a decoded view cannot be reported as a span
    the user can see, and the explainability requirement fails silently.
    """
    report = normalize(text)
    for view in report.views:
        assert len(view.offset_map) == len(view.text)
        for off in view.offset_map:
            assert 0 <= off <= len(report.original)


@PROFILE
@given(ANY_TEXT)
def test_canonical_normalisation_is_idempotent(text: str) -> None:
    """Normalising twice must equal normalising once.

    A non-idempotent normaliser means the canonical form depends on how many
    times it was applied, which makes rule authoring and corpus preprocessing
    subtly inconsistent.
    """
    once = normalize(text).canonical.text
    twice = normalize(once).canonical.text
    assert once == twice


@PROFILE
@given(CLEAN_ASCII)
def test_printable_ascii_is_a_fixed_point_of_canonical(text: str) -> None:
    """The false-positive guard, stated as a property.

    Printable ASCII contains no confusables, no zero-width characters and no
    bidi controls. The canonical view must return it unchanged - otherwise the
    normaliser is manufacturing signal from ordinary English.
    """
    report = normalize(text)
    assert report.canonical.text == text


@PROFILE
@given(ANY_TEXT)
def test_decode_depth_never_exceeds_cap(text: str) -> None:
    report = normalize(text)
    assert report.max_decode_depth <= 6


@PROFILE
@given(ANY_TEXT)
def test_canonical_view_is_always_present(text: str) -> None:
    report = normalize(text)
    assert report.view(ViewKind.CANONICAL) is not None
    assert report.canonical in report.views


@PROFILE
@given(ANY_TEXT)
def test_spans_are_well_formed(text: str) -> None:
    report = normalize(text)
    for t in report.transforms:
        if t.span is not None:
            start, end = t.span
            assert 0 <= start <= end <= len(report.original)


@PROFILE
@given(ANY_TEXT)
def test_obfuscation_score_is_a_probability_like_scalar(text: str) -> None:
    report = normalize(text)
    assert 0.0 <= report.obfuscation_score <= 1.0


# --------------------------------------------------------------------------
# Latency regression guard.
# --------------------------------------------------------------------------


def test_l0_stays_inside_its_share_of_the_tier_a_budget() -> None:
    """L0 must leave room for L1 inside the 5ms Tier-A budget (ADR-0002).

    This test exists because the first implementation missed it by 2.2x: the
    ROT-N decoder ran a per-character Python loop across 25 candidate
    rotations, on every request, including plain English that cannot be
    ROT-encoded. Measured p95 was 10.9ms for clean 2KB input.

    The threshold is 2.5ms - half the Tier-A budget - not the measured value.
    A tighter bound would fail on a loaded CI runner and teach everyone to
    ignore it; a looser one would not catch a regression of the kind that
    prompted it.
    """
    import time

    text = ("Please summarise the attached quarterly report and highlight risks. " * 31)[:2048]
    for _ in range(20):
        normalize(text)

    latencies: list[float] = []
    for _ in range(200):
        started = time.perf_counter()
        normalize(text)
        latencies.append((time.perf_counter() - started) * 1000.0)

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < 2.5, f"L0 p95 {p95:.2f}ms exceeds its share of the Tier-A budget"
