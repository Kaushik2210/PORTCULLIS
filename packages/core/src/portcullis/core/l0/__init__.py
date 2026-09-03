"""L0 - normalisation and de-obfuscation.

Runs before every other layer, unconditionally. A classifier trained on clean
text is bypassed by base64 or a single Cyrillic character, so normalising after
scoring would be useless.

Two ideas carry the design:

* **Views, not one string.** Canonical normalisation is near-lossless and safe
  to trust. Leetspeak and spaced-letter folding are lossy and would create
  false positives on hashes, IDs and ordinary English, so they live in a
  separate FOLDED view that a rule engine can weight down. Decoded payloads are
  a third kind again.
* **Obfuscation is signal.** Heavy encoding is reported, not just undone -
  decode depth and transform mix feed fusion as features in their own right.
"""

from .confusables import CONFUSABLES
from .decode import MAX_DECODE_DEPTH, Decoded, decode_layered
from .normalize import DEFAULT_CONFIG, MAX_INPUT_CHARS, NormalizerConfig, normalize
from .types import NormalizationReport, TextView, Transform, ViewKind

__all__ = [
    "CONFUSABLES",
    "DEFAULT_CONFIG",
    "MAX_DECODE_DEPTH",
    "MAX_INPUT_CHARS",
    "Decoded",
    "NormalizationReport",
    "NormalizerConfig",
    "TextView",
    "Transform",
    "ViewKind",
    "decode_layered",
    "normalize",
]
