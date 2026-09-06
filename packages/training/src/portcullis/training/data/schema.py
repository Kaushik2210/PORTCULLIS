"""The common row shape every source is normalised into.

Every Tier-1 source (ADR-0004) ships its own column names and its own label
vocabulary. Everything downstream - dedup, splitting, the audit, the manifest -
works against this one shape instead of three.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Row:
    """One normalised example.

    `family` is the disjointness key (ADR-0004, spec: "disjoint by attack
    family and source, not by row"). These corpora carry no fine-grained
    attack-taxonomy labels, only binary injection/benign - so at this stage
    `family` is the source dataset. That is a documented limitation, not an
    oversight: see docs/DATASET_CARD.md.
    """

    id: str
    text: str
    label: int  # 0 = benign, 1 = attack
    source: str
    source_split: str
    license: str
    tier: int
    family: str

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"{self.id}: label must be 0 or 1, got {self.label}")
        if self.tier not in (1, 2):
            raise ValueError(f"{self.id}: tier must be 1 or 2, got {self.tier}")
