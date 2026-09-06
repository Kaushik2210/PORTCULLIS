"""ONNX export, INT8 quantisation, accuracy-delta check, latency histogram.

Reuses the export/quantise approach proven at Milestone 0
(packages/eval/.../bench_l2_frontier.py) against the *actual trained*
checkpoint this time, not a randomly-initialised head - M0 measured latency
only and said so explicitly; this is where that number gets an accuracy
claim attached to it, and the spec requires both directions reported
(before/after latency, and the accuracy delta from quantising).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .labels import TAXONOMY
from .train import MAX_SEQ_LEN, load_training_config

_BENIGN_INDEX = TAXONOMY.index("benign")

# onnxruntime ships no type stubs (pyproject.toml override); named explicitly
# rather than left as a bare Any so a reader can see this is a deliberate
# untyped boundary, matching the same pattern used for datasketch in dedup.py.
OrtSession = Any

# Tier B/C from ADR-0002: typical (~512B / seq128) and worst-case (2KB / seq512).
SEQ_LENS = (128, 512)
WARMUP_ITERS = 20
MIN_ITERS = 50
BUDGET_S = 20.0


def export_onnx(checkpoint_dir: Path, out_path: Path, seq_len: int = MAX_SEQ_LEN) -> None:
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    dummy_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )


def quantize_int8(src: Path, dst: Path) -> None:
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def _onnx_benign_logit(sess: OrtSession, ids: np.ndarray, mask: np.ndarray) -> float:
    (logits,) = sess.run(None, {"input_ids": ids, "attention_mask": mask})
    return float(logits[0][_BENIGN_INDEX])


def accuracy_delta(
    fp32_path: Path,
    int8_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    true_labels: list[int],
    *,
    max_seq_len: int,
) -> dict[str, object]:
    """Compares fp32 vs INT8 predictions on the same inputs: agreement rate
    on the binary attack/benign call, and each variant's own accuracy
    against ground truth - the spec's "verify accuracy delta is within
    tolerance" made concrete rather than asserted.

    `max_seq_len` must match what the checkpoint was trained at
    (load_training_config): truncating validation text differently from
    training would make this comparison unfair to both variants equally,
    but still not a fair test of what the model actually learned.
    """
    fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    agree, fp32_correct, int8_correct = 0, 0, 0
    for text, true_label in zip(texts, true_labels, strict=True):
        enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="np")
        ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

        fp32_attack = _onnx_benign_logit(fp32, ids, mask) < 0  # benign logit < 0 => attack-leaning
        int8_attack = _onnx_benign_logit(int8, ids, mask) < 0

        agree += int(fp32_attack == int8_attack)
        fp32_correct += int(fp32_attack == bool(true_label))
        int8_correct += int(int8_attack == bool(true_label))

    n = len(texts)
    return {
        "n": n,
        "fp32_int8_agreement_rate": agree / n,
        "fp32_accuracy": fp32_correct / n,
        "int8_accuracy": int8_correct / n,
        "accuracy_delta": (int8_correct - fp32_correct) / n,
    }


@dataclass(frozen=True, slots=True)
class LatencyStats:
    n: int
    p50: float
    p95: float
    p99: float
    raw: tuple[float, ...]

    def as_report_dict(self) -> dict[str, float]:
        """The persisted view - everything except the raw samples, which
        exist only to feed the histogram plot and would bloat the JSON."""
        return {"n": self.n, "p50": self.p50, "p95": self.p95, "p99": self.p99}


def _bench_one(path: Path, seq_len: int, threads: int) -> LatencyStats:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0xC0FFEE)
    feed = {
        "input_ids": rng.integers(0, 1000, size=(1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }
    for _ in range(WARMUP_ITERS):
        sess.run(None, feed)

    latencies: list[float] = []
    started = time.perf_counter()
    while len(latencies) < MIN_ITERS or (
        time.perf_counter() - started < BUDGET_S and len(latencies) < 500
    ):
        t0 = time.perf_counter()
        sess.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return LatencyStats(
        n=len(latencies),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        raw=tuple(latencies),
    )


def latency_histogram(int8_path: Path, out_path: Path) -> dict[str, LatencyStats]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # SEQ_LENS always holds Tier B and Tier C (ADR-0002) - two entries by
    # construction, so plt.subplots always returns an Axes array here, never
    # the single-Axes case a len==1 guard would exist to handle.
    fig, axes = plt.subplots(1, len(SEQ_LENS), figsize=(6 * len(SEQ_LENS), 4.5))

    results: dict[str, LatencyStats] = {}
    for ax, seq_len in zip(axes, SEQ_LENS, strict=True):
        stats = _bench_one(int8_path, seq_len, threads=4)
        results[f"seq{seq_len}"] = stats
        ax.hist(stats.raw, bins=30, color="#4C72B0", alpha=0.85)
        ax.axvline(stats.p50, color="green", linestyle="--", label=f"p50={stats.p50:.1f}ms")
        ax.axvline(stats.p95, color="orange", linestyle="--", label=f"p95={stats.p95:.1f}ms")
        ax.axvline(stats.p99, color="red", linestyle="--", label=f"p99={stats.p99:.1f}ms")
        ax.set_title(f"L2 INT8 latency, seq={seq_len} (n={stats.n})")
        ax.set_xlabel("latency (ms)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    return results


def run(*, checkpoint_dir: Path, data_dir: Path, model_out_dir: Path, report_out_dir: Path) -> None:
    config = load_training_config(checkpoint_dir)
    max_seq_len = config["max_seq_len"]
    fp32_path = model_out_dir / "l2.onnx"
    int8_path = model_out_dir / "l2.int8.onnx"

    print(f"exporting ONNX (seq={max_seq_len}) from {checkpoint_dir}", flush=True)
    export_onnx(checkpoint_dir, fp32_path, seq_len=max_seq_len)
    print("quantising INT8", flush=True)
    quantize_int8(fp32_path, int8_path)
    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"  fp32: {fp32_mb:.1f}MB, int8: {int8_mb:.1f}MB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    rows = []
    with (data_dir / "validation.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    true_labels = [r["label"] for r in rows]

    print(f"measuring accuracy delta on {len(texts)} validation rows", flush=True)
    delta = accuracy_delta(
        fp32_path, int8_path, tokenizer, texts, true_labels, max_seq_len=max_seq_len
    )
    print(
        f"  fp32 acc={delta['fp32_accuracy']:.3f} int8 acc={delta['int8_accuracy']:.3f} "
        f"delta={delta['accuracy_delta']:+.3f} agreement={delta['fp32_int8_agreement_rate']:.3f}",
        flush=True,
    )

    print("measuring INT8 latency histogram", flush=True)
    latency = latency_histogram(int8_path, report_out_dir / "l2-latency-histogram.png")
    for seq, stats in latency.items():
        print(
            f"  {seq}: p50={stats.p50:.1f}ms p95={stats.p95:.1f}ms p99={stats.p99:.1f}ms",
            flush=True,
        )

    report_out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "fp32_size_mb": round(fp32_mb, 1),
        "int8_size_mb": round(int8_mb, 1),
        "accuracy_delta": delta,
        "latency": {k: v.as_report_dict() for k, v in latency.items()},
    }
    (report_out_dir / "l2-export-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote {report_out_dir / 'l2-export-report.json'}", flush=True)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--model-out-dir", type=Path, required=True)
    ap.add_argument("--report-out-dir", type=Path, default=Path("docs/benchmarks"))
    args = ap.parse_args()
    run(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        model_out_dir=args.model_out_dir,
        report_out_dir=args.report_out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
