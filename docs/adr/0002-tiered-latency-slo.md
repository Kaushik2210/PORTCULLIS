# ADR-0002 — Tiered, measurement-derived latency SLO

**Status:** Accepted
**Date:** Milestone 0
**Evidence:** [`docs/benchmarks/l2-frontier.md`](../benchmarks/l2-frontier.md) — 48 measured cells,
regenerable with `just bench`.

## Context

The original requirement was a single figure:

> L0–L2 detection must run at **p95 < 25ms on CPU** for a 2KB prompt.

A 2KB prompt is roughly 500 tokens, so this asks a transformer encoder to complete a seq-512
forward pass, plus normalisation and rule evaluation, inside 25ms on CPU. That figure was set
before anything was measured. Milestone 0 measured it.

## Evidence

Four candidate encoders, exported to ONNX and dynamically quantised to INT8, measured on the
development machine (i7-1360P, 12C/16T hybrid, Windows, onnxruntime 1.29.0). **p95 ms, INT8:**

| Model | INT8 MB | seq 128 / 4thr | seq 256 / 4thr | seq 512 / 4thr | seq 512 / 1thr |
|---|---:|---:|---:|---:|---:|
| **MiniLM-L6** | 22 | **21.2** | 44.7 | **135.4** | 194.1 |
| DistilRoBERTa-base | 79 | 34.8 | 125.6 | 347.6 | 432.6 |
| DeBERTa-v3-small | 164 | 97.3 | 145.0 | 310.3 | 652.5 |
| ModernBERT-base | 144 | 109.8 | 324.9 | 1051.8 | 1475.8 |

Findings:

1. **At seq 512, no candidate comes within 5x of 25ms.** The fastest model at best-case
   threading is 135.4ms. ModernBERT-base — the model the spec named first — is 1051.8ms, off by
   a factor of 42.
2. **At seq 128, exactly one cell in the 48-cell sweep fits:** MiniLM-L6 INT8 at 4 threads,
   21.2ms p95.
3. **Thread count roughly halves or doubles everything.** A request served while the gateway is
   saturated gets closer to one core, and the seq-512 leader degrades from 135.4ms to 194.1ms.
   An SLO quoted from the idle-host number would not survive load.
4. **File size is not a latency proxy.** DeBERTa-v3-small is the largest artifact (164MB) yet
   beats ModernBERT-base (144MB) at every sequence length; its size is a 128k-token embedding
   matrix, which costs disk rather than compute.

### The numbers were validated before being used

The first sweep produced physically impossible results — seq 128 timing slower than seq 256, and
a 22M-parameter model timing slower than an 82M one. Root cause: the ONNX artifacts were written
inside a OneDrive-synced directory and were being uploaded during the run. Artifacts were moved
off the sync root, warmup raised 5→20, the sample floor 30→50, and the harness now flags its own
suspect cells (drift beyond ±15% between run halves, low n, or p95 > 2.5x p50) and excludes them
from the frontier plot.

A second question remained: were the absolute numbers real, or an artifact of the export? A
PyTorch-eager reference at seq 512 / 4 threads gives **1168.9ms eager vs 1222.8ms ONNX — a ratio
of 1.05x**. A pathological graph would show 3–10x. It does not; the hardware is simply this slow
for a model of this class. This rules out the export tooling as the cause. It does not rule out
a limitation shared by both paths, and this ADR claims no more than that.

## Decision

**Replace the single 25ms figure with a three-tier SLO, each tier attached to a traffic class.**

| Tier | Path | Input | Budget (p95) | Basis |
|---|---|---|---|---|
| **A** | L0 + L1 short-circuit | 2KB | **< 10ms** | Measured at M2: 6.7ms. Revised from 5ms - see below. |
| **B** | Full ingress incl. L2 | typical (≤512B, seq 128) | **< 25ms** | Measured: 21.2ms, MiniLM-L6 INT8 @ 4thr |
| **C** | Full ingress incl. L2 | worst case (2KB, seq 512) | **< 150ms** | Measured: 135.4ms, same configuration |

### Tier A revised from 5ms to 10ms (Milestone 2)

The 5ms figure was labelled above as a target, not a measurement. Milestone 2 measured it
against the real 50-rule corpus on 2KB of clean English:

| Component | p50 | p95 |
|---|---:|---:|
| L0 normalisation | 0.47ms | 0.98ms |
| L1, 43 user-scoped rules | 3.36ms | 5.76ms |
| **L0 + L1** | **3.8ms** | **~6.7ms** |

L1 dominates, at roughly 80-260us per rule. The cost is inherent to the rule shapes: an
alternation of negation verbs followed by a bounded gap has no literal prefix for the regex
engine to skip ahead on, so each rule scans the full input.

**A literal prefilter was implemented and then removed.** Deriving each rule's required literals
by inspecting its pattern text is not sound: `\[INST\]` is an escaped literal bracket rather than
a character class, so stripping character classes discarded a required literal and the rule was
silently skipped on input it should have matched. It also turned out *slower*, because a
per-rule substring sweep over 2KB costs more than the regex it was avoiding. **In a security
product, a false negative introduced by a latency optimisation is a strictly worse outcome than
the milliseconds it buys**, so the budget moves instead of the detection.

The budget moves to 10ms rather than to the measured 6.7ms, to absorb CI-runner variance without
the guard test becoming flaky and being ignored.

**Why this does not weaken the cascade argument.** The cost case in ADR-0001 does not depend on
Tier A being any particular absolute number - it depends on the *ratio* between the
short-circuit path and the full path. At 6.7ms versus 135ms that ratio is roughly 20x, and the
argument for running cheap layers first is unchanged. A reviewer should hold the architecture to
the ratio and to the measured short-circuit rate (Milestone 9), not to this figure.

**The sound optimisation, recorded as future work.** Author-declared literals: an explicit
`requires: [ignore, disregard, ...]` field per rule, checked as a set intersection against the
tokenised input before the regex runs, and validated by a test asserting that every one of the
rule's own positive cases contains at least one declared literal. That is sound by assertion
plus verification rather than by inference, and it is reviewable in the YAML. It is deferred
because it needs a positive case per alternation branch to be trustworthy, which is a rule-corpus
convention to establish rather than a code change to make.

Supporting decisions:

1. **ModernBERT-base is eliminated as an L2 candidate.** It is the slowest of the four at every
   sequence length while offering no latency argument in return. It may still be revisited if
   Milestone 4 shows an accuracy gap large enough to justify Tier C moving, but the burden of
   proof sits with it.
2. **Final L2 selection is deferred to Milestone 4** and made on the latency/TPR frontier, not on
   latency alone. This ADR fixes the latency axis only; the accuracy axis does not exist yet, and
   choosing a model now would be choosing on half the evidence.
3. **Tier A is load-bearing for the whole architecture.** The cascade's cost argument (ADR-0001)
   assumes most traffic never reaches L2. If L0+L1 cannot hold 5ms, or if the short-circuit rate
   is low, the cascade is paying for complexity it does not recover. The eval harness must report
   per-layer traffic share, and a failure here reopens ADR-0001.
4. **The SLO is quoted at 4 intra-op threads, with 1-thread figures published alongside.**
   Concurrency planning uses the 1-thread column. ONNX Runtime thread counts are pinned
   explicitly in deployment; the default (all cores) does not survive multiple concurrent
   requests.

## Alternatives rejected

**Keep the hard 25ms at 2KB.** Would require an L2 roughly 5x faster than the fastest candidate
measured. Nothing in the tested field is close, and the remaining levers (aggressive truncation,
a sub-MiniLM model) trade away exactly the detection capability L2 exists to provide.

**Raise the single budget to 150ms.** Simple and honest, but it prices every request at the
worst case and discards the cascade's central claim — that cost should scale with ambiguity
rather than volume. Tiering keeps that claim measurable.

**Report only the idle-host number.** Rejected as misleading. The 4-thread figure is real but
describes a condition a loaded gateway is not in.

## Consequences

- Three budgets to track instead of one, and the eval harness must classify each request by
  input size and cascade depth to attribute it to a tier.
- Tier C at 150ms is a genuine cost for large inputs. Mitigations to evaluate at M4/M5: capping
  L2's max sequence length with L1 covering the remainder, sliding-window scoring, and ORT
  transformer-specific graph optimisation (untested — the 1.05x eager ratio suggests headroom
  the generic optimiser is not capturing).
- The published numbers are specific to this CPU. `just bench` regenerates them anywhere, and
  the host block is recorded alongside every table so a reader knows what hardware produced them.
- **These are latency-only measurements on randomly-initialised heads.** No accuracy claim is
  made or implied, and none may be quoted from this ADR.
