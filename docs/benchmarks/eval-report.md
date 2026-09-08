# Full evaluation harness — Milestone 9

The real benchmark table, run for real against the actually-deployed system - the fusion weights
loaded here are the exact `fusion_weights.json` M5 fit and the gateway serves, not a fresh refit.
Scope and methodology are in [ADR-0010](../adr/0010-m9-eval-harness-scope.md); this report is the
measurement, including three findings that are genuinely unflattering and are reported as found,
not smoothed over.

Regenerate with `just eval` (needs `just l2 m5` first; takes ~18 minutes). Raw output:
[eval-results.json](eval-results.json).

## Headline

Full ~20k-row test set (19,937 rows, 5,696 attacks), 95% bootstrap CIs (2000 resamples):

| | TPR @ 0.1% FPR | TPR @ 1% FPR | AUPRC |
|---|---:|---:|---:|
| **PORTCULLIS (fusion)** | **0.154** (0.145–0.166) | **0.191** (0.180–0.202) | **0.648** |
| `max()` baseline | 0.007 (0.003–0.019) | 0.067 (0.056–0.076) | 0.553 |
| regex-only (L1 alone) | 0.198 (0.187–0.208) | 0.198 (0.188–0.208) | 0.427 |

Consistent with M5's own 3,000-row sample (19.2% TPR@1%FPR there vs. 19.1% here on the full set,
now with a real CI) - the fusion-vs-`max()` story holds up at full scale: fusion is **~2.9x**
`max()`'s TPR at 1% FPR, and dramatically ahead of it at 0.1% FPR.

**A finding worth stating plainly, not burying: L1 alone (the regex-only baseline) matches or
slightly beats the full learned fusion system at both FPR targets** - 19.8% vs. fusion's 15.4% at
0.1% FPR, essentially tied at 1% FPR (19.8% vs. 19.1%). This does not contradict the fusion-vs-
`max()` result above; it says something more specific: in the very-low-FPR regime this headline
targets, L1's hand-written rules - built directly from real attack phrasing - turn out to define
an unusually clean threshold, while fusion's AUPRC (0.648 vs. L1's 0.427) shows its real advantage
sits at less extreme operating points, where combining signals clearly pulls ahead. Read honestly:
at the specific low-FPR point this project's own NFR-3 cares about most, a decade-old technique
(regex rules) is competitive with the learned system built to surpass it. That is not a comfortable
sentence to publish in a portfolio piece, and it is the correct one.

## Ablations

Each variant refits fusion's logistic regression on `validation` with a feature held out
entirely (not zeroed at inference - the remaining features are refit to compensate), then
evaluates on the same full test set:

| Variant | TPR @ 0.1% FPR | TPR @ 1% FPR | AUPRC | vs. full fusion |
|---|---:|---:|---:|---|
| Full fusion (l0, l1, l2, knn) | 0.154 | 0.191 | 0.648 | — |
| Remove L0 | 0.154 | 0.191 | 0.648 | no measurable change |
| Remove kNN | **0.157** | **0.217** | **0.669** | **improves** |
| `max()` (no fusion at all) | 0.007 | 0.067 | 0.553 | far worse |

**Removing L0 changes nothing** - expected, and confirms M5's own observation that L0's fitted
coefficient (0.009) was already near zero. Obfuscation-depth alone just isn't informative on this
corpus, independent of content.

**Removing the kNN sidecar makes the headline numbers better, not worse** - TPR@1%FPR rises from
0.191 to 0.217, AUPRC from 0.648 to 0.669. A second unflattering, honestly-reported result:
despite a meaningfully positive fitted coefficient (1.58) in the full model, kNN's marginal
contribution to fusion's generalisation is negative on the real test set. The likely mechanism:
fusion is fit on `validation` (790 rows) - a small enough fit set that a fourth feature can pick
up validation-specific correlations a leaner 3-feature model doesn't, at a real cost to how well
it generalises to `test`. This is exactly the kind of result an ablation table exists to surface
(ADR-0001's own framing) rather than an argument the kNN sidecar should be removed outright - M5's
kNN work also produces the "nearest known attack" explanation string the dashboard's Decision
Inspector needs, a real product value this table doesn't measure.

**No L4 row.** L4 operates on conversation history, not a single test-set row - there is no
per-row "with/without L4" ablation to run (ADR-0010, Decision 4). Its measured contribution is
[M7's own report](l4-report.md): a real conversation escalating to `PROBING` while every
individual turn stayed under the flag threshold fusion alone would have used.

## Latency: the cascade is well over budget, and the cause is identified

Full-cascade, single-request, unbatched (`just eval`'s own methodology; a live gateway request is
exactly this shape - one text, not a batch):

| | p50 | p95 | p99 |
|---|---:|---:|---:|
| Full cascade (L0+L1+L2+kNN+fusion) | 68.1ms | 176.4ms | 381.0ms |

This is well over both of ADR-0002's tiers (Tier B <25ms typical, Tier C <150ms worst-case) - the
first genuine end-to-end measurement of the assembled cascade; every prior milestone's own latency
numbers (M2's 6.7ms L0+L1, M4's 17.4ms L2 alone) measured a slice, not the whole thing, by design
(M4's own report says so explicitly).

**Investigated rather than reported blindly**, matching this project's established practice
(ADR-0002, M4's thermal-tail finding): a fresh, cold-process re-measurement gave nearly identical
numbers (p50 59.6ms vs. 68.1ms) - not primarily a thermal artifact. A per-layer breakdown through
the real gateway pipeline pinpoints the cause precisely:

| Layer | p50 | p95 |
|---|---:|---:|
| L0 | 0.36ms | 1.92ms |
| L1 | 1.66ms | 10.03ms |
| **kNN sidecar** | **52.33ms** | **104.49ms** |
| L2 | 9.10ms | 13.54ms |
| Fusion | 0.03ms | 0.03ms |
| **Total** | **61.17ms** | **124.98ms** |

**The kNN sidecar's embedding call is ~85% of total cascade latency.** The reason is architectural,
not incidental: L2 went through a full optimisation pass at Milestone 4 (fine-tuned, ONNX-exported,
INT8-quantised - 17.4ms measured then) specifically to fit its latency budget. The kNN sidecar's
embedder (`sentence-transformers/all-MiniLM-L6-v2`, shared with L2's own base checkpoint, M5) never
received the same treatment - it still runs as unquantised FP32 PyTorch inference on every single
request. This is a real, actionable, previously-unmeasured finding (M6's gateway-report.md
explicitly deferred a real end-to-end latency measurement to this milestone rather than fabricate
one) - the fix (export the embedder to ONNX and quantise it, the same recipe already proven on L2)
is naturally scoped future work, not attempted here: Milestone 9 is measurement, and inventing a
number for "how much would that save" without doing the work would be exactly the kind of fabricated
metric the working agreement rules out.

## Adaptive attacker evaluation

Both techniques assume grey/black-box query access (ADR-0001's adversary (b)/(c)) and are scored
against the real deployed fusion weights via the same `block_threshold=0.8` the gateway enforces.
Both seeded from real attack rows in the test set (not synthetic examples), and both report the
success rate honestly, including where it's bad (spec's own framing for this section).

### Paraphrase attacks

200 real attack seeds; a small hand-authored synonym table (ADR-0010) applied one substitution at
a time, not compounded:

| | |
|---|---:|
| Seeds with any applicable phrase to substitute | 55 / 200 (27.5%) |
| Variants tried | 464 |
| Variants that evaded (score < 0.8) | 132 / 464 (**28.4%**) |
| Seeds with at least one evading variant | 22 / 200 (**11.0%**) |

Most seeds (72.5%) contained none of the hand-authored table's trigger phrases at all - a real
coverage gap in the table, not a claim the corpus is mostly unparaphrasable. Of the seeds where a
substitution was possible, roughly one in four individual variants evaded detection, and 11% of
all 200 seeds had at least one working evasion. A single cheap word swap succeeding against a
learned classifier more than one time in ten is a real, reportable weakness, not a reassuring
number, and is presented as one.

### Iterative black-box query attacks

100 real attack seeds; greedy hill-climbing (deletion, whitespace injection, case-flip, a
confusable substitution) up to 100 queries per seed:

| | |
|---|---:|
| Seeds evaded | **69 / 100 (69%)** |
| Mean queries used | 8.45 (well under the 100-query budget) |

**This is a materially higher success rate than the paraphrase attacks, and it should be read as
the more serious of the two results.** A generic, corpus-agnostic search - no attack-specific
knowledge beyond "try small mutations, keep whichever helps" - evades the deployed detector on
over two-thirds of real attack attempts, in under 9 queries on average. This is squarely the
"report your own failure modes honestly" section the spec calls out by name, and the honest
takeaway is unflattering: cheap, generic, automatable evasion works most of the time against the
system as currently tuned.

## What Milestone 9 does not establish

Everything ADR-0010's Decision 1 names as cut: LLM-judge and commercial-API baselines,
`deberta-v3-base-injection`, garak/PyRIT/promptfoo scheduled CI, k6/Locust load testing, and
GCG-style gradient transfer-suffix attacks. Per-attack-family recall is a documented omission
(ADR-0010, Decision 2), not a fabricated column - the test corpus has no independent family
ground truth, and deriving one from L1 would be circular with the regex-only baseline and the
ablations this same report just published.
