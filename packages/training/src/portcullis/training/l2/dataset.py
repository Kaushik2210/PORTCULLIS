"""Assembles masked multi-label training examples (ADR-0005).

No row is ever excluded from training outright. A row weak-labelled against
the L1 taxonomy (labels.py) supervises all 8 output dimensions. A row the
weak-labeller could not place (an attack matching no L1 rule) still
supervises the `benign` dimension, which is known false from the trustworthy
binary ground truth - it is only the 7 family dimensions that go unsupervised
for that row, via a per-element loss mask rather than by dropping the row.

This is what keeps L2's binary detection signal decoupled from weak-label
noise: every one of the ~16k train rows contributes to "is this benign or
not," while only the ~48% of attack rows L1 could place contribute to "which
family." See ADR-0005 for why that split exists and what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .labels import TAXONOMY

_BENIGN_INDEX = TAXONOMY.index("benign")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    id: str
    text: str
    target: tuple[float, ...]
    mask: tuple[float, ...]


def build_training_examples(
    binary_rows: list[dict[str, object]],
    weak_rows: list[dict[str, object]],
) -> list[TrainingExample]:
    """`binary_rows` is the full partition (every row, binary label only,
    e.g. loaded from train.jsonl). `weak_rows` is the output of
    weak_label_corpus.py for the same partition - a subset, keyed by id.
    """
    weak_by_id = {str(r["id"]): r for r in weak_rows}
    examples: list[TrainingExample] = []

    for row in binary_rows:
        row_id = str(row["id"])
        weak = weak_by_id.get(row_id)

        if weak is not None:
            labels = weak["labels"]
            assert isinstance(labels, list)
            target = tuple(float(v) for v in labels)
            mask = (1.0,) * len(TAXONOMY)
        else:
            # Excluded from family supervision (label must be 1: benign rows
            # are never excluded by weak_label_row's own contract). Known
            # false on the benign dimension; everything else unsupervised.
            target_list = [0.0] * len(TAXONOMY)
            mask_list = [0.0] * len(TAXONOMY)
            mask_list[_BENIGN_INDEX] = 1.0
            target = tuple(target_list)
            mask = tuple(mask_list)

        examples.append(TrainingExample(id=row_id, text=str(row["text"]), target=target, mask=mask))

    return examples
