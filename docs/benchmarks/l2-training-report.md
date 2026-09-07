# L2 training, calibration and export — Milestone 4

Real numbers from a real run, not a projection. Scope was fixed before this run started
(see [ADR-0005](../adr/0005-l2-weak-labels-and-model-choice.md)): MiniLM-L6, 1 epoch, prove the
pipeline end-to-end rather than chase accuracy. Everything below is what that scope actually
produced.

## Training

| | |
|---|---:|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (22M params) |
| Objective | Masked multi-label BCE, 8 classes (ADR-0005) |
| Train rows | 16,030 (full partition; every row supervises the benign bit, 52% also supervise a family — [weak-label-coverage.md](weak-label-coverage.md)) |
| Sequence length | 128 (measured throughput-optimal on this CPU; see below) |
| Batch size | 32 |
| Epochs | 1 |
| Wall clock | 1942s (~32 min), CPU only — no GPU on this machine (ADR-0002) |
| Train loss | 0.69 → 0.052 |
| Validation masked BCE | 0.0463 |

Loss dropped smoothly and validation loss came in *below* final train loss — no sign of
overfitting on a single epoch, which is expected: one pass over 16k rows is closer to
"the model has seen the data once" than to convergence.

**Throughput note.** The 256-token default took ~83 min/epoch on this CPU (3.2 rows/s); 128
tokens at batch 32 measured 8.3 rows/s, ~33 min/epoch. That tuning pass is recorded in the
Milestone-4 commit history, not just asserted here.

**Process note.** The first attempt at this run was killed mid-epoch (step 150/501) when its
host session ended, with no checkpoint saved — `train.py` only checkpointed at the very end at
that point. Fixed by adding interim checkpointing every 200 steps plus after each epoch, before
relaunching. The run reported here is the second, successful attempt.

## Calibration (NFR-2)

Fit against **validation's real binary labels**, never the weak multi-label targets - weak-label
noise (ADR-0005) must not leak into the number this project uses to claim "calibrated."

| | Before (raw sigmoid) | After (Platt-scaled) |
|---|---:|---:|
| Brier score | 0.0124 | 0.0103 |

Platt parameters: `a=1.7646, b=0.7625` (positive slope — calibrated probability rises with the
raw score, as it must). Both Brier scores are already low; the raw output was reasonably
calibrated before scaling, plausibly a side effect of training with `BCEWithLogitsLoss` in the
first place. Reliability diagram: [reliability-diagram.png](reliability-diagram.png).

## ONNX export and INT8 quantisation

| | |
|---|---:|
| FP32 size | 86.8 MB |
| INT8 size | 22.0 MB (4.0x reduction) |

**Accuracy delta** (790 validation rows, raw 0.5 threshold — not the fixed-FPR operating point
Milestone 9 will use for the headline metric, a different question):

| | FP32 | INT8 |
|---|---:|---:|
| Accuracy | 98.61% | 98.35% |
| Agreement between the two | | 99.24% |
| Delta | | **-0.25 points** |

Quantisation cost almost nothing here. `-0.25pp` is well inside any reasonable tolerance.

## Latency (INT8, this CPU)

| Tier | seq | p50 | p95 | p99 | n |
|---|---:|---:|---:|---:|---:|
| B (typical) | 128 | 12.6ms | **17.4ms** | 20.8ms | 500 |
| C (worst case) | 512 | 147.2ms | **283.7ms** | 313.2ms | 122 |

Histogram: [l2-latency-histogram.png](l2-latency-histogram.png). Full report:
[l2-export-report.json](l2-export-report.json).

**A number worth being honest about.** The first latency measurement, taken immediately after
the 32-minute training run finished, showed p95 of 54.8ms (seq128) and 322.7ms (seq512) - 2.4-2.6x
worse than this table. That was investigated rather than reported: a standalone re-measurement
minutes later, with the CPU no longer under sustained load, came back close to this table's
numbers. This is the same hybrid P/E-core thermal-tail behaviour ADR-0002 already documented for
this machine, reasserting itself - not a new bug, and not something to silently average away.
The seq128 tier came back clean and stable (17.4ms, actually *better* than ADR-0002's cold
reference of 21.2ms). The seq512 tier's p95/p99 still show more spread than ADR-0002's reference
(283.7ms vs. 135.4ms there) - plausibly residual thermal state from training immediately
beforehand, plausibly this machine's known tail instability being more visible at the more
compute-intensive sequence length. Compared against ADR-0002's own Tier B/C figures - which, like this table, report L2's own
latency as the benchmark point rather than a literal L0+L1+L2 sum: **Tier B (<25ms) is now met**
(17.4ms vs. the ADR's own 21.2ms reference - the trained model is faster than M0's random-init
estimate at this sequence length). **Tier C (<150ms) is not met** (283.7ms vs. budget, roughly
1.9x over) - not a surprise given M0 already flagged Tier C as the tier requiring further work
(sequence-length capping, sliding-window scoring, or ORT graph optimisation), and this is the
first time it has been checked against a real trained model rather than a random-init one.

**What this comparison does not include, honestly:** a true combined L0+L1+L2 measurement.
Tier A's 6.7ms figure is from Milestone 2, against the 50-rule corpus at that point; four rules
were added during this milestone's weak-label investigation (ADR-0005), so L1's real current
cost is not re-measured here. Adding a stale L0+L1 number to this run's genuine L2 number would
produce a figure precise-looking enough to be mistaken for measured when it is not. A real
end-to-end cascade latency check is undone work, naturally suited to Milestone 9's harness.

## What this milestone does and does not establish

Does: proves the full pipeline runs end-to-end on real data with real numbers - weak-label,
train, calibrate, export, measure - and produces genuine calibration and latency artifacts, not
placeholders.

Does not: establish that this is a well-trained model. One epoch on a corpus where 52% of attack
rows carry no family label (ADR-0005) is a pipeline proof, not a claim about detection quality.
TPR@FPR, the project's actual headline metric, is explicitly Milestone 9's job and remains `TBD`
in the README. The three-model latency/TPR frontier the original spec asked for is still
deferred, per the scope decision at the start of this milestone.
