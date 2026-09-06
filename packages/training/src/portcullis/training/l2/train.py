"""Fine-tunes the L2 candidate encoder on the masked multi-label objective.

Model: sentence-transformers/all-MiniLM-L6-v2 (ADR-0005 - the M0 latency
leader; the three-model frontier is deferred, not silently dropped).

Runs on CPU: there is no GPU on this machine (established at Milestone 0).
Checkpoints are written outside the repo entirely, not just gitignored
inside it - Milestone 0's benchmark spike corrupted its own measurements by
writing large artifacts into a OneDrive-synced folder that got uploaded
mid-run, and the fix there was the same as here: write outside any sync
root by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .dataset import TrainingExample, build_training_examples
from .labels import TAXONOMY

MODEL_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
NUM_LABELS = len(TAXONOMY)
MAX_SEQ_LEN = 256
DEFAULT_OUT_DIR = Path.home() / "AppData" / "Local" / "portcullis" / "l2_checkpoint"


class MaskedMultiLabelDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(ex.target, dtype=torch.float32),
            "mask": torch.tensor(ex.mask, dtype=torch.float32),
        }


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy, averaged only over unmasked (target, logit)
    pairs. mask == 0 means "unknown for this row" (ADR-0005's excluded
    family bits), not "target is negative" - it must not contribute to the
    gradient at all, which is why this is not simply a zero target.
    """
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    masked = per_element * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def load_training_config(checkpoint_dir: Path) -> dict[str, int]:
    """Read back the actual hyperparameters a checkpoint was trained with.

    Consumed by calibrate.py and export.py so they tokenize at the real
    max_seq_len rather than assuming this module's default - see the note
    where training_config.json is written in train().
    """
    path = checkpoint_dir / "training_config.json"
    if not path.exists():
        return {"max_seq_len": MAX_SEQ_LEN}
    result: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_partition(data_dir: Path, name: str) -> list[TrainingExample]:
    binary_rows: list[dict[str, object]] = []
    with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                binary_rows.append(json.loads(line))

    weak_path = data_dir / f"{name}.weak.jsonl"
    weak_rows: list[dict[str, object]] = []
    if weak_path.exists():
        with weak_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    weak_rows.append(json.loads(line))
    return build_training_examples(binary_rows, weak_rows)


def evaluate(
    model: PreTrainedModel,
    loader: DataLoader[dict[str, torch.Tensor]],
) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = masked_bce_loss(out.logits, batch["target"], batch["mask"])
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(
    *,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    log_every: int,
    max_seq_len: int = MAX_SEQ_LEN,
    max_train_rows: int | None = None,
) -> None:
    torch.manual_seed(0)

    print(f"loading tokenizer/model: {MODEL_CHECKPOINT}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    )
    model.train()

    print("loading train/validation partitions", flush=True)
    train_examples = load_partition(data_dir, "train")
    val_examples = load_partition(data_dir, "validation")
    if max_train_rows is not None and max_train_rows < len(train_examples):
        # A deterministic *stride*, not a prefix: train.jsonl is written
        # sorted by row id (pipeline.py), which is the original PromptShield
        # row order - M3 found that corpus is heavily templated, and template
        # variants often sit in contiguous blocks. A prefix could massively
        # over- or under-represent whichever templates happen to sit first.
        # Striding through the full range is far safer against that ordering
        # artefact, and still fully reproducible (NFR-6): same cap, same
        # subset, every run.
        stride = len(train_examples) / max_train_rows
        train_examples = [train_examples[int(i * stride)] for i in range(max_train_rows)]
    print(f"  train: {len(train_examples)} rows, validation: {len(val_examples)} rows", flush=True)

    train_ds = MaskedMultiLabelDataset(train_examples, tokenizer, max_seq_len)
    val_ds = MaskedMultiLabelDataset(val_examples, tokenizer, max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        running_loss, seen = 0.0, 0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = masked_bce_loss(out.logits, batch["target"], batch["mask"])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            seen += 1
            if step % log_every == 0:
                elapsed = time.perf_counter() - epoch_start
                rate = step * batch_size / elapsed
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running_loss / seen:.4f} ({rate:.1f} rows/s)",
                    flush=True,
                )
                running_loss, seen = 0.0, 0

        val_loss = evaluate(model, val_loader)
        epoch_time = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "val_loss": val_loss, "epoch_seconds": epoch_time})
        print(
            f"epoch {epoch} done in {epoch_time:.0f}s - validation masked BCE: {val_loss:.4f}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "taxonomy.json").write_text(json.dumps(list(TAXONOMY)), encoding="utf-8")
    # Downstream steps (calibrate.py, export.py) must tokenize at the same
    # max_seq_len the model was actually trained at, not the module default -
    # this file's own MAX_SEQ_LEN constant is only the default, and Milestone
    # 4's real run overrides it (128, not 256) for CPU throughput. Persisting
    # it here is what lets those steps read the truth instead of assuming it.
    (out_dir / "training_config.json").write_text(
        json.dumps({"max_seq_len": max_seq_len, "epochs": epochs, "batch_size": batch_size}),
        encoding="utf-8",
    )
    print(f"\nsaved checkpoint to {out_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Deterministically stride-sample train to this many rows (see train() docstring).",
    )
    args = ap.parse_args()
    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        max_seq_len=args.max_seq_len,
        max_train_rows=args.max_train_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
