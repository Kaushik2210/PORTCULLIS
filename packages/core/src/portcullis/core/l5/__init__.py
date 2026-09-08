"""L5 - egress inspection. Detection that only looks at input is half a
product: canary-token leak detection (zero-FP by construction), secret/PII
egress scanning, and markdown-image/fabricated-tool-call exfiltration
detection - all on the model's *response*, not the user's request
(ADR-0009).
"""

from .canary import generate_canary, inject_into_system_prompt, scan_for_canary
from .exfil import scan_for_fabricated_tool_calls, scan_for_markdown_image_exfil
from .scanner import redact, scan
from .secrets import scan_for_secrets
from .stream_scanner import DEFAULT_WINDOW_CHARS, SlidingWindowScanner, StreamScanResult
from .types import EgressFinding, EgressScanResult, FindingKind, mask_preview

__all__ = [
    "DEFAULT_WINDOW_CHARS",
    "EgressFinding",
    "EgressScanResult",
    "FindingKind",
    "SlidingWindowScanner",
    "StreamScanResult",
    "generate_canary",
    "inject_into_system_prompt",
    "mask_preview",
    "redact",
    "scan",
    "scan_for_canary",
    "scan_for_fabricated_tool_calls",
    "scan_for_markdown_image_exfil",
    "scan_for_secrets",
]
