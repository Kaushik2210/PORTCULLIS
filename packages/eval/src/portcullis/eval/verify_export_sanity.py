"""Sanity-check the ONNX frontier numbers against a PyTorch-eager reference.

The Milestone-0 sweep produced absolute latencies well below the throughput this
model class is usually reported at. Before those numbers are used to set an SLO
(ADR-0002), they need one question answered:

    Is the model genuinely this slow on this CPU, or did the export produce a
    pathological graph?

The discriminating test is a plain PyTorch eager forward pass on identical input.

* ONNX materially **slower** than eager  -> the export or the ORT graph is the
  problem, and the frontier numbers understate what the model can do. The ADR
  must not quote them as the model's floor.
* ONNX at parity with or **faster** than eager -> the export is fine and the
  hardware is simply this slow. The frontier numbers stand.

Reports the ratio and states which conclusion the evidence supports, rather than
leaving the reader to infer it.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification

NUM_LABELS = 8


def time_eager(hf_id: str, seq_len: int, threads: int, iters: int) -> list[float]:
    torch.set_num_threads(threads)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=NUM_LABELS)
    model.eval()

    ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    mask = torch.ones((1, seq_len), dtype=torch.long)

    lat: list[float] = []
    with torch.no_grad():
        for _ in range(5):
            model(input_ids=ids, attention_mask=mask)
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def time_onnx(onnx_path: Path, seq_len: int, threads: int, iters: int) -> list[float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(0xC0FFEE)
    feed: dict[str, Any] = {
        "input_ids": rng.integers(0, 1000, size=(1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }

    lat: list[float] = []
    for _ in range(5):
        sess.run(None, feed)
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-id", default="answerdotai/ModernBERT-base")
    ap.add_argument("--onnx", type=Path, default=Path("models/bench/ModernBERT-base.onnx"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    if not args.onnx.exists():
        print(f"missing {args.onnx} - run `just bench` first")
        return 1

    print(f"reference check: {args.hf_id}  seq={args.seq_len} threads={args.threads}\n")

    eager = time_eager(args.hf_id, args.seq_len, args.threads, args.iters)
    onnx = time_onnx(args.onnx, args.seq_len, args.threads, args.iters)

    e_med = statistics.median(eager)
    o_med = statistics.median(onnx)
    ratio = o_med / e_med

    print(f"  PyTorch eager (fp32) : {e_med:8.2f} ms  median of {len(eager)}")
    print(f"  ONNX Runtime  (fp32) : {o_med:8.2f} ms  median of {len(onnx)}")
    print(f"  ratio onnx/eager     : {ratio:8.2f}x\n")

    if ratio > 1.25:
        print("  VERDICT: ONNX is materially slower than eager.")
        print("  The exported graph is the bottleneck, not the model or the CPU.")
        print("  Frontier numbers understate the models - do not quote them as a floor.")
    elif ratio < 0.8:
        print("  VERDICT: ONNX is faster than eager, as expected.")
        print("  Export is healthy; the frontier numbers reflect the model on this CPU.")
    else:
        print("  VERDICT: ONNX and eager are at parity.")
        print("  No evidence of a pathological export; this CPU is simply this slow")
        print("  for a model of this size. Frontier numbers stand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
