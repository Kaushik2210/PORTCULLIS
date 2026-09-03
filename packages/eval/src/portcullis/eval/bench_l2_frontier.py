"""Milestone-0 latency spike: measure the L2 candidate frontier on *this* CPU.

This script exists to set the L2 latency budget from measurement rather than from
assumption (ADR-0002). It measures **latency only** — the classification heads are
randomly initialised and no accuracy claim is made or implied here.

Design notes that matter for reading the numbers:

* **Two thread settings are reported, and they answer different questions.**
  ``threads=4`` is best-case single-request latency on an idle box. ``threads=1``
  is what a request actually gets when the gateway is saturated and every core is
  serving someone else. A gateway SLO quoted from the idle number is a fiction;
  both are published.
* **p95 needs samples.** Each config runs until it hits either ``--max-iters`` or a
  per-config time budget, with a floor of 30 iterations. The realised N is recorded
  next to every percentile so a reader can judge the estimate.
* **Thermal drift is exposed, not hidden.** The target is a 13th-gen mobile CPU with
  P/E cores; sustained load throttles. Each config reports the median of its first
  half against its second half. A large gap means the number is not steady-state.

Run via ``just bench``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification

# Candidate encoders for L2. Multi-label head (8 taxonomy classes, ADR-0001).
CANDIDATES: dict[str, str] = {
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    "DeBERTa-v3-small": "microsoft/deberta-v3-small",
    "DistilRoBERTa-base": "distilbert/distilroberta-base",
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
}

NUM_LABELS = 8
SEQ_LENS = (128, 256, 512)
THREAD_COUNTS = (1, 4)
MIN_ITERS = 50
WARMUP_ITERS = 20

# ONNX artifacts are multi-hundred-MB and must NOT live inside a synced folder
# (OneDrive/Dropbox). A sync client uploading them mid-run competes for CPU and
# I/O and produces measurements that are physically impossible - shorter inputs
# timing slower than longer ones. Default outside the repo entirely.
DEFAULT_WORKDIR = Path.home() / "AppData" / "Local" / "portcullis" / "bench"


@dataclass(frozen=True)
class Measurement:
    """One (model, variant, seq_len, threads) latency cell."""

    model: str
    variant: str
    seq_len: int
    threads: int
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    drift_ratio: float
    size_mb: float
    suspect: str

    @property
    def trustworthy(self) -> bool:
        return not self.suspect


@dataclass(frozen=True)
class ExportFailure:
    model: str
    stage: str
    error: str


DRIFT_TOLERANCE = 0.15  # +/-15% between run halves before a cell is suspect


def _diagnose(lat: list[float], drift: float) -> str:
    """Flag a cell whose own numbers say it was measured under contention.

    A benchmark that cannot tell you when it was disturbed will happily publish
    a physically impossible frontier. Cheap to compute, and it turns a silent
    data-quality failure into a visible one.
    """
    reasons: list[str] = []
    if abs(drift - 1.0) > DRIFT_TOLERANCE:
        reasons.append(f"drift={drift:.2f}")
    if len(lat) < MIN_ITERS:
        reasons.append(f"n={len(lat)}<{MIN_ITERS}")
    if lat and _percentile(lat, 95) > 2.5 * _percentile(lat, 50):
        reasons.append("p95>2.5x-p50")
    return ",".join(reasons)


def _percentile(xs: list[float], q: float) -> float:
    ordered = sorted(xs)
    idx = min(round(q / 100.0 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def export_onnx(hf_id: str, out_path: Path, seq_len: int = 512) -> None:
    """Export a sequence-classification encoder to ONNX with dynamic axes.

    Uses the legacy TorchScript exporter explicitly: the dynamo path is still
    uneven across encoder architectures, and this script must not silently
    measure a different graph for different models.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=NUM_LABELS, torch_dtype=torch.float32
    )
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
    """Dynamic INT8 quantisation via ONNX Runtime's own quantiser.

    Deliberately avoids a dependency on `optimum`: `quantize_dynamic` ships with
    onnxruntime and covers the weight-only INT8 case this spike needs.
    """
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def benchmark(
    onnx_path: Path,
    seq_len: int,
    threads: int,
    max_iters: int,
    budget_s: float,
) -> tuple[list[float], float]:
    """Return (per-iteration latencies in ms, realised wall seconds)."""
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

    for _ in range(WARMUP_ITERS):  # arena allocation, graph setup, clock ramp
        sess.run(None, feed)

    latencies: list[float] = []
    started = time.perf_counter()
    while len(latencies) < max_iters:
        t0 = time.perf_counter()
        sess.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if len(latencies) >= MIN_ITERS and (time.perf_counter() - started) > budget_s:
            break
    return latencies, time.perf_counter() - started


def measure_all(
    workdir: Path, max_iters: int, budget_s: float
) -> tuple[list[Measurement], list[ExportFailure]]:
    results: list[Measurement] = []
    failures: list[ExportFailure] = []

    for label, hf_id in CANDIDATES.items():
        print(f"\n=== {label} ({hf_id}) ===", flush=True)
        fp32 = workdir / f"{label}.onnx"
        int8 = workdir / f"{label}.int8.onnx"

        try:
            if not fp32.exists():
                print("  exporting fp32 ...", flush=True)
                export_onnx(hf_id, fp32)
        except Exception as exc:  # report the failure, never abort the sweep
            print(f"  EXPORT FAILED: {exc}", flush=True)
            failures.append(ExportFailure(label, "onnx-export", str(exc)[:400]))
            continue

        try:
            if not int8.exists():
                print("  quantising int8 ...", flush=True)
                quantize_int8(fp32, int8)
        except Exception as exc:
            print(f"  QUANTISE FAILED: {exc}", flush=True)
            failures.append(ExportFailure(label, "int8-quantise", str(exc)[:400]))

        for variant, path in (("fp32", fp32), ("int8", int8)):
            if not path.exists():
                continue
            size_mb = path.stat().st_size / (1024 * 1024)
            for seq_len in SEQ_LENS:
                for threads in THREAD_COUNTS:
                    lat, _ = benchmark(path, seq_len, threads, max_iters, budget_s)
                    half = len(lat) // 2
                    first = statistics.median(lat[:half]) if half else float("nan")
                    second = statistics.median(lat[half:]) if half else float("nan")
                    m = Measurement(
                        model=label,
                        variant=variant,
                        seq_len=seq_len,
                        threads=threads,
                        n=len(lat),
                        p50_ms=round(_percentile(lat, 50), 2),
                        p95_ms=round(_percentile(lat, 95), 2),
                        p99_ms=round(_percentile(lat, 99), 2),
                        mean_ms=round(statistics.fmean(lat), 2),
                        drift_ratio=round(second / first, 3) if first else float("nan"),
                        size_mb=round(size_mb, 1),
                        suspect=_diagnose(lat, second / first if first else 1.0),
                    )
                    results.append(m)
                    print(
                        f"  {variant:>4} seq={seq_len:>3} thr={threads} "
                        f"n={m.n:>4} p50={m.p50_ms:>7.2f} p95={m.p95_ms:>7.2f} "
                        f"p99={m.p99_ms:>7.2f} drift={m.drift_ratio}",
                        flush=True,
                    )
    return results, failures


def host_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_markdown(
    results: list[Measurement], failures: list[ExportFailure], host: dict[str, str], out: Path
) -> None:
    lines: list[str] = [
        "# L2 candidate latency frontier",
        "",
        "Generated by `just bench` — do not hand-edit.",
        "",
        "**Latency only.** Classification heads are randomly initialised; this table",
        "makes no accuracy claim. Model choice is made on the latency/TPR frontier once",
        "Milestone 4 supplies the accuracy axis.",
        "",
        "## Host",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | `{v}` |" for k, v in host.items()]
    lines += [
        "",
        "## Measurements",
        "",
        "`threads=1` is the per-request budget under saturation; `threads=4` is best-case",
        "on an idle host. `drift` is second-half median over first-half median — values",
        "far from 1.0 indicate thermal throttling rather than a steady-state number.",
        "",
        "| model | variant | seq | thr | n | p50 ms | p95 ms | p99 ms | drift | MB | suspect |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.variant} | {r.seq_len} | {r.threads} | {r.n} | "
            f"{r.p50_ms} | **{r.p95_ms}** | {r.p99_ms} | {r.drift_ratio} | {r.size_mb} | "
            f"{r.suspect or 'ok'} |"
        )
    if failures:
        lines += ["", "## Export failures", "", "| model | stage | error |", "|---|---|---|"]
        lines += [f"| {f.model} | {f.stage} | `{f.error}` |" for f in failures]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(results: list[Measurement], out: Path) -> None:
    """Frontier plot: p95 latency vs model size, INT8, seq=512."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable - skipping plot", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, threads in zip(axes, THREAD_COUNTS, strict=True):
        pts = [r for r in results if r.variant == "int8" and r.threads == threads and r.trustworthy]
        for seq, marker in zip(SEQ_LENS, ("o", "s", "^"), strict=True):
            sub = sorted((r for r in pts if r.seq_len == seq), key=lambda r: r.size_mb)
            if not sub:
                continue
            ax.plot(
                [r.size_mb for r in sub],
                [r.p95_ms for r in sub],
                marker=marker,
                linestyle="--",
                label=f"seq={seq}",
            )
            for r in sub:
                ax.annotate(
                    r.model,
                    (r.size_mb, r.p95_ms),
                    fontsize=7,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax.set_title(f"INT8, intra_op_threads={threads}")
        ax.set_xlabel("ONNX model size (MB)")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("p95 latency (ms)")
    fig.suptitle("L2 candidate frontier - p95 latency vs model size (measured, CPU)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--outdir", type=Path, default=Path("docs/benchmarks"))
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--budget-s", type=float, default=25.0)
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    results, failures = measure_all(args.workdir, args.max_iters, args.budget_s)
    host = host_info()

    payload = {
        "host": host,
        "results": [asdict(r) for r in results],
        "failures": [asdict(f) for f in failures],
    }
    (args.outdir / "l2-frontier.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(results, failures, host, args.outdir / "l2-frontier.md")
    write_plot(results, args.outdir / "l2-frontier.png")
    print(f"\n{len(results)} measurements, {len(failures)} failures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
