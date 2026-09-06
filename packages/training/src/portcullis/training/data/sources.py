"""The source registry. Every corpus PORTCULLIS trains or evaluates on is
listed here, once, with its licence and tier - nothing gets ingested that
isn't declared in this file first.

Column mappings and label semantics below were verified against a live pull of
each dataset at Milestone 3, not assumed from documentation:

    PromptShield          'prompt' / 'label'  (0/1, int)   train+validation+test
    deepset/prompt-inj.   'text'   / 'label'  (0/1, int)   train+test
    jackhhao/jailbreak    'prompt' / 'type'   ('benign'/'jailbreak', str)  train+test

Label semantics (1 = attack) were spot-checked against actual row text for
deepset, not inferred from the dataset card - see the Milestone-3 commit for
the verification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    tier: int
    license: str
    text_column: str
    label_column: str
    label_map: Mapping[object, int]
    hf_id: str | None = None
    hf_splits: tuple[str, ...] = ()
    note: str = ""

    def map_label(self, raw: object) -> int:
        try:
            return self.label_map[raw]
        except KeyError as exc:
            raise ValueError(
                f"{self.name}: unrecognised label {raw!r}; expected one of "
                f"{sorted(map(repr, self.label_map))}"
            ) from exc


PROMPT_SHIELD = SourceSpec(
    name="promptshield",
    tier=1,
    license="apache-2.0",
    hf_id="hendzh/PromptShield",
    hf_splits=("train", "validation", "test"),
    text_column="prompt",
    label_column="label",
    label_map={0: 0, 1: 1},
    note="Backbone corpus (ADR-0004). Own train/validation/test respected as "
    "the split skeleton; near-dup remediation only removes leakage, it does "
    "not reshuffle the split.",
)

DEEPSET_PROMPT_INJECTIONS = SourceSpec(
    name="deepset_prompt_injections",
    tier=1,
    license="apache-2.0",
    hf_id="deepset/prompt-injections",
    hf_splits=("train", "test"),
    text_column="text",
    label_column="label",
    label_map={0: 0, 1: 1},
    note="Supplementary cross-source eval only (ADR-0004) - too small "
    "(662 rows) to anchor a TPR@0.1%FPR estimate. Never enters train.",
)

JACKHHAO_JAILBREAK = SourceSpec(
    name="jackhhao_jailbreak",
    tier=1,
    license="apache-2.0",
    hf_id="jackhhao/jailbreak-classification",
    hf_splits=("train", "test"),
    text_column="prompt",
    label_column="type",
    label_map={"benign": 0, "jailbreak": 1},
    note="Supplementary cross-source eval only (ADR-0004), same reasoning as deepset above.",
)

HARD_NEGATIVES = SourceSpec(
    name="hard_negatives",
    tier=1,
    license="apache-2.0",  # this project's own licence; see NOTICE
    hf_id=None,
    text_column="text",
    label_column="label",
    label_map={0: 0, 1: 1},
    note="Authored in-repo (resources/hard_negatives.jsonl), not downloaded. "
    "Eval-only by construction - it exists to measure false positives, so "
    "training on it would be circular.",
)

TIER1_SOURCES: tuple[SourceSpec, ...] = (
    PROMPT_SHIELD,
    DEEPSET_PROMPT_INJECTIONS,
    JACKHHAO_JAILBREAK,
    HARD_NEGATIVES,
)

# Tier 2: gated, opt-in, excluded from the default reproducible path (ADR-0004).
# Registered for completeness; not exercised by this pipeline without
# --include-gated, and even then requires an accepted HF licence click-through
# and HF_TOKEN, which is not something this repo can automate for a user.
LMSYS_CHAT_1M = SourceSpec(
    name="lmsys_chat_1m",
    tier=2,
    license="LMSYS-Chat-1M Dataset License Agreement (gated, click-through, "
    "includes a right-to-require-deletion clause)",
    hf_id="lmsys/lmsys-chat-1m",
    hf_splits=("train",),
    text_column="conversation",
    label_column="",  # unlabelled; all-benign by construction if ever used
    label_map={},
    note="NOT fetched by this pipeline. Requires an authenticated HF account "
    "that has accepted the licence agreement. See ADR-0004.",
)

TIER2_SOURCES: tuple[SourceSpec, ...] = (LMSYS_CHAT_1M,)
