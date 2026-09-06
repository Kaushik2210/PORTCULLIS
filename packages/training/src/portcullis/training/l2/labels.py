"""Weak multi-label supervision for L2, derived from the L1 rule taxonomy.

The problem this solves: the Tier-1 corpus (ADR-0004) carries binary
ground truth only (attack/benign) - PromptShield and its supplements never
say *which kind* of attack a row is. L2 is specified as an 8-class
multi-label classifier (7 attack families + benign), and no labelled
multi-class source exists to train it on.

The fix, and its cost, are both stated plainly rather than glossed over
(ADR-0005 has the full account):

* An attack row's binary label is trustworthy - it came from the source
  corpus, not from us. Which *family* it belongs to is not directly known,
  so we ask L1's own rules, which is the only labelled signal available.
* This makes L2's multi-label training data a function of L1's own
  coverage. A genuinely novel attack shape that no L1 rule recognises
  cannot be weak-labelled, so it is excluded from training rather than
  guessed at - see `weak_label_row`'s return contract below. The practical
  consequence is that L2, trained this way, is *not* independently
  verified to generalise to attack families L1 cannot already name. That is
  a real limitation of this label source, not a property of the model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from portcullis.core.l0 import NormalizationReport
from portcullis.core.l1 import RuleEngine, Scope


class LabelledText(Protocol):
    """What weak_label_row needs from a row, duck-typed rather than imported.

    Matches portcullis.training.data.schema.Row's shape without creating a
    dependency edge back onto the data package - weak-labelling is a
    consumer of rows, not a part of producing them. Declared via read-only
    properties, not plain attributes: Row is a frozen dataclass, and a plain
    attribute in a Protocol asserts a *settable* field, which a frozen
    dataclass structurally is not.
    """

    @property
    def text(self) -> str: ...

    @property
    def label(self) -> int: ...


# Fixed order matters: training targets are positional vectors, and this
# order must stay stable across every script that reads or writes one.
# Matches the seven labels actually used across rules/*.yaml plus "benign".
TAXONOMY: tuple[str, ...] = (
    "direct_override",
    "role_play_jailbreak",
    "system_prompt_extraction",
    "data_exfiltration",
    "tool_abuse",
    "encoding_evasion",
    "indirect_injection",
    "benign",
)

_BENIGN_INDEX = TAXONOMY.index("benign")
_LABEL_INDEX = {name: i for i, name in enumerate(TAXONOMY)}

Normalizer = Callable[[str], NormalizationReport]


def weak_label_row(
    row: LabelledText,
    engine: RuleEngine,
    normalize: Normalizer,
    *,
    scope: Scope = Scope.USER,
) -> tuple[int, ...] | None:
    """Return an 8-dim multi-hot label vector, or None to exclude the row.

    Contract:
    * label == 0 (benign): always labelled, only the benign bit set. Binary
      benign ground truth needs no corroboration from L1.
    * label == 1 (attack): multi-hot over every taxonomy label any L1 rule
      matched, scanned at `scope`. If no rule matched, returns None - the
      caller must exclude the row from multi-label training rather than
      inventing a label. Excluding is the documented behaviour, not a bug;
      see the module docstring and ADR-0005.
    """
    if row.label == 0:
        vec = [0] * len(TAXONOMY)
        vec[_BENIGN_INDEX] = 1
        return tuple(vec)

    report = normalize(row.text)
    matches = engine.scan(report, scope)
    fired_labels = {label for m in matches for label in m.labels if label in _LABEL_INDEX}

    if not fired_labels:
        return None

    vec = [0] * len(TAXONOMY)
    for label in fired_labels:
        vec[_LABEL_INDEX[label]] = 1
    return tuple(vec)
