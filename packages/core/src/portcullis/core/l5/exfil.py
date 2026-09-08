"""Markdown-image exfiltration and fabricated tool-call syntax - both
under-tested vectors the spec calls out specifically. Coarse, explainable
heuristics in the same spirit as L1's rules (ADR-0001): a pattern with a
stated reason, not a model, and not claimed to be complete.
"""

from __future__ import annotations

import re

from .entropy import shannon_entropy
from .types import EgressFinding, FindingKind, mask_preview

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_QUERY_VALUE = re.compile(r"[?&][^=&]+=([^&\s]+)")
_EXFIL_QUERY_MIN_LEN = 16
_EXFIL_QUERY_ENTROPY = 3.5
"""Lower than secrets.py's threshold (4.0): a query parameter carrying
exfiltrated data doesn't need to look like a cryptographic key to be a real
leak - plain scraped text run through URL-encoding already reads as fairly
high entropy. Reasoned default, not tuned (ADR-0009)."""

_TOOL_CALL_MARKERS = re.compile(
    r"<(?:tool_call|function_call|tool_use)\b|"
    r'\{\s*"(?:name|function|tool_name)"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:'
)


def scan_for_markdown_image_exfil(text: str) -> tuple[EgressFinding, ...]:
    """A markdown image whose URL carries a long, high-entropy query value
    is the classic exfil shape - the renderer fetches the URL as a side
    effect of displaying the response, so the "click" the attacker needs
    never has to happen (spec's own framing)."""
    findings: list[EgressFinding] = []
    for image_match in _MARKDOWN_IMAGE.finditer(text):
        url = image_match.group(1)
        for query_match in _QUERY_VALUE.finditer(url):
            value = query_match.group(1)
            if len(value) < _EXFIL_QUERY_MIN_LEN:
                continue
            if shannon_entropy(value) >= _EXFIL_QUERY_ENTROPY:
                findings.append(
                    EgressFinding(
                        kind=FindingKind.EXFIL,
                        rule="markdown-image-exfil",
                        span=image_match.span(),
                        preview=mask_preview(image_match.group()),
                        rationale=(
                            "markdown-image-exfil: an image URL with a long, high-entropy "
                            "query value - a renderer fetches this automatically, so no click "
                            "is required for the query value to reach an external host."
                        ),
                    )
                )
                break  # one finding per image is enough to explain the block
    return tuple(findings)


def scan_for_fabricated_tool_calls(text: str) -> tuple[EgressFinding, ...]:
    """Tool-call-shaped syntax appearing in ordinary assistant prose is
    suspicious regardless of whether it's a real call: either the model is
    fabricating a call it was never asked to make, or echoing an
    injected instruction back verbatim. Coarse on purpose - a marker
    pattern, not a parser - since the goal is flagging the shape, not
    validating a real tool-call protocol."""
    findings: list[EgressFinding] = []
    for m in _TOOL_CALL_MARKERS.finditer(text):
        findings.append(
            EgressFinding(
                kind=FindingKind.EXFIL,
                rule="fabricated-tool-call",
                span=m.span(),
                preview=mask_preview(m.group()),
                rationale=(
                    "fabricated-tool-call: tool/function-call-shaped syntax in free-text "
                    "output - either an unrequested fabricated call or an injected "
                    "instruction echoed back verbatim."
                ),
            )
        )
    return tuple(findings)
