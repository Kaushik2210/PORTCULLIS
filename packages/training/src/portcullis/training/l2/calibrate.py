"""Platt scaling + reliability diagram (NFR-2: calibrated output, not an
arbitrary risk score - proved with a diagram, not asserted).

Calibration is fit and measured against the **trustworthy binary label**
from validation.jsonl, never against weak multi-label targets. Weak labels
are a training-time convenience with a documented noise ceiling (ADR-0005);
letting that noise leak into the number this project uses to claim
"calibrated" would undermine the one thing NFR-2 exists to guarantee.

The model's raw signal is reduced to one scalar before calibration: the
`benign` logit, negated. That is the natural P(attack)-shaped score L2
contributes to fusion (ADR-0001) - a single number per request, not an
8-vector - and it is also the one dimension every training row supervised
(ADR-0005), so it is the most reliably trained of the eight outputs.

Platt scaling is stored as two floats (a, b: calibrated = sigmoid(a*raw + b))
rather than a pickled sklearn object, so the serving path never needs
scikit-learn as a runtime dependency - just two numbers and a sigmoid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .labels import TAXONOMY
from .train import load_training_config

_BENIGN_INDEX = TAXONOMY.index("benign")


@dataclass(frozen=True, slots=True)
class PlattParams:
    a: float
    b: float

    def calibrate(self, raw_score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * raw_score + self.b)))


def raw_attack_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_seq_len: int,
    batch_size: int = 32,
) -> np.ndarray:
    """-logit(benign) for each text: higher means more attack-like.

    Deliberately the pre-sigmoid logit, not sigmoid(benign)-inverted -
    Platt scaling is defined over an unbounded decision-function value, and
    fitting it on an already-squashed [0,1] score double-compresses the tails
    exactly where a fixed-FPR threshold decision is most sensitive.

    `max_seq_len` is required, not defaulted: it must match what the
    checkpoint was actually trained at (load_training_config), not this
    module's own idea of a sensible default.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, truncation=True, max_length=max_seq_len, padding=True, return_tensors="pt"
            )
            logits = model(**enc).logits
            scores.append((-logits[:, _BENIGN_INDEX]).numpy())
    return np.concatenate(scores)


def fit_platt(raw_scores: np.ndarray, true_labels: np.ndarray) -> PlattParams:
    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(raw_scores.reshape(-1, 1), true_labels)
    return PlattParams(a=float(clf.coef_[0][0]), b=float(clf.intercept_[0]))


def reliability_diagram(
    raw_scores: np.ndarray,
    true_labels: np.ndarray,
    platt: PlattParams,
    out_path: Path,
    *,
    n_bins: int = 10,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uncalibrated = 1.0 / (1.0 + np.exp(-raw_scores))
    calibrated = platt.calibrate(raw_scores)

    frac_pos_before, mean_pred_before = calibration_curve(
        true_labels, uncalibrated, n_bins=n_bins, strategy="quantile"
    )
    frac_pos_after, mean_pred_after = calibration_curve(
        true_labels, calibrated, n_bins=n_bins, strategy="quantile"
    )

    brier_before = float(np.mean((uncalibrated - true_labels) ** 2))
    brier_after = float(np.mean((calibrated - true_labels) ** 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated", linewidth=1)
    ax.plot(mean_pred_before, frac_pos_before, "o-", label=f"raw (Brier={brier_before:.3f})")
    ax.plot(mean_pred_after, frac_pos_after, "s-", label=f"Platt-scaled (Brier={brier_after:.3f})")
    ax.set_xlabel("mean predicted P(attack)")
    ax.set_ylabel("observed fraction attack")
    ax.set_title("L2 reliability diagram (validation, binary ground truth)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)

    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "n_bins": n_bins,
        "n_examples": len(true_labels),
    }


def run(*, checkpoint_dir: Path, data_dir: Path, out_dir: Path) -> None:
    config = load_training_config(checkpoint_dir)
    max_seq_len = config["max_seq_len"]
    print(f"loading checkpoint from {checkpoint_dir} (max_seq_len={max_seq_len})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    rows = []
    with (data_dir / "validation.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    true_labels = np.array([r["label"] for r in rows], dtype=np.float64)
    print(f"scoring {len(texts)} validation rows (true binary labels)", flush=True)

    raw = raw_attack_scores(model, tokenizer, texts, max_seq_len=max_seq_len)
    platt = fit_platt(raw, true_labels)
    print(f"Platt params: a={platt.a:.4f} b={platt.b:.4f}", flush=True)

    stats = reliability_diagram(raw, true_labels, platt, out_dir / "reliability-diagram.png")
    print(f"Brier score: {stats['brier_before']:.4f} -> {stats['brier_after']:.4f}", flush=True)

    calib_path = checkpoint_dir / "calibration.json"
    calib_path.write_text(
        json.dumps({"platt_a": platt.a, "platt_b": platt.b, **stats}, indent=2), encoding="utf-8"
    )
    print(f"wrote {calib_path}", flush=True)
    print(f"wrote {out_dir / 'reliability-diagram.png'}", flush=True)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/benchmarks"))
    args = ap.parse_args()
    run(checkpoint_dir=args.checkpoint_dir, data_dir=args.data_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
