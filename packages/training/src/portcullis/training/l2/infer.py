"""A reusable L2 scorer: ONNX INT8 checkpoint + Platt calibration -> a single
calibrated P(attack) per text.

Extracted as its own module because two different M5 consumers need exactly
this (fusion-fitting scores the corpus through L2; a future gateway will
score live requests through it) and duplicating the ONNX-session and
calibration-application logic between them would be the same bug waiting to
happen twice - export.py's accuracy_delta already has one inline copy of the
"run ONNX, read the benign logit" pattern; this is the version other code
reuses instead of copying that pattern a third time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .calibrate import PlattParams
from .labels import TAXONOMY
from .train import load_training_config

_BENIGN_INDEX = TAXONOMY.index("benign")

OrtSession = Any


class L2Scorer:
    """Loads once, scores many. Construction does the (relatively) expensive
    work - session creation, tokenizer load, calibration file read - so a
    fusion-fitting pass over thousands of rows pays that cost once, not
    once per row.
    """

    def __init__(
        self,
        onnx_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        platt: PlattParams,
        max_seq_len: int,
    ) -> None:
        self._session: OrtSession = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizer
        self._platt = platt
        self._max_seq_len = max_seq_len

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path, onnx_path: Path) -> L2Scorer:
        config = load_training_config(checkpoint_dir)
        calib_path = checkpoint_dir / "calibration.json"
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        platt = PlattParams(a=calib["platt_a"], b=calib["platt_b"])
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        return cls(onnx_path, tokenizer, platt, config["max_seq_len"])

    def score(self, text: str) -> float:
        """Calibrated P(attack) in [0, 1]."""
        enc = self._tokenizer(
            text, truncation=True, max_length=self._max_seq_len, return_tensors="np"
        )
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        (logits,) = self._session.run(None, {"input_ids": ids, "attention_mask": mask})
        raw = -float(logits[0][_BENIGN_INDEX])  # higher = more attack-like
        calibrated = self._platt.calibrate(np.array([raw]))
        return float(calibrated[0])
