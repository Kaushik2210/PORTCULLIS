# kNN sidecar, learned fusion, and policy engine — Milestone 5

Real numbers from a real run. Design rationale for all three pieces built this milestone is in
[ADR-0006](../adr/0006-knn-fusion-and-policy.md); this report is the measurement side of that
ADR.

## kNN sidecar

| | |
|---|---:|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (shared with L2, ADR-0005) |
| Index | Brute-force cosine similarity, numpy (ADR-0003 resolution) |
| Indexed rows | 6,831 (all `label == 1` rows in `train.jsonl`) |
| Query latency | 0.69ms single-query, 0.26ms/query batched (6,831 × 384-dim) |

**Performance bug found and fixed this milestone.** The first scoring run through the sidecar
embedded one query text at a time and measured 2.1 rows/s — on track for a ~30 minute run to
score the fusion fit+eval sets. Corpus-embedding (6,831 rows in one batched call, at index-build
time) had already run at ~42 rows/s; per-row scoring simply wasn't using batching. Fixed by
adding `KnnScorer.score_batch()` and refactoring `fusion/fit.py._score_all()` to batch-embed
every row's text up front, before the per-row L0/L1/L2 loop. Re-measured from the actual fit run:

```
scoring validation (fusion fit set)
  batch-embedding 790 rows for the kNN sidecar
  scored 500/790 (141.6 rows/s)
scoring test sample (fusion eval set, n=3000)
  batch-embedding 3000 rows for the kNN sidecar
  scored 3000/3000 (58.8 rows/s)
```

67x faster on validation. The test-sample run is slower per-row (58.8 vs 141.6 rows/s) because
L2's ONNX forward pass, not embedding, is now the bottleneck once embedding is no longer paying
the per-call tax — consistent with M4's own measured 12.6ms/row L2 cost.

## Fusion

Logistic regression over four features (`l0`, `l1`, `l2`, `knn`) — chosen over a gradient-boosted
tree so the fitted coefficients themselves double as the human-readable "fusion arithmetic" the
dashboard's Decision Inspector needs (NFR-5, see ADR-0006 for the full argument).

**Split discipline.** Fit on `validation` (790 rows); evaluated on a deterministic-stride sample
of `test` (3,000 of 19,937 rows). Not fit on `train` — L2 itself was fit on train (0.052 final
loss, M4), so train rows aren't a fair test of what fusion should learn to trust about L2's
score. `validation` was previously only spent on L2's two-parameter Platt calibration, so it's
still an uncontaminated set for fusion's four coefficients.

**Fitted weights:**

| Layer | Coefficient |
|---|---:|
| Intercept | -4.4484 |
| L0 (obfuscation) | 0.0090 |
| L1 (rules) | 2.2509 |
| L2 (classifier) | 6.4159 |
| kNN (nearest-attack) | 1.5783 |

L2 dominates, L1 and kNN both meaningfully positive, L0 near zero — obfuscation alone is a weak
standalone signal here, which matches ADR-0001's framing of L0 as a feature that matters combined
with content, not in isolation (an all-ASCII "ignore previous instructions" scores zero on L0 and
should still be caught).

**Fusion vs. `max()` baseline**, evaluated on the 3,000-row test sample:

| | TPR @ 1% FPR | AUPRC |
|---|---:|---:|
| Learned fusion | **0.1918** | **0.6400** |
| `max()` baseline | 0.0619 | 0.5448 |
| Delta | **+0.1300** | **+0.0952** |

Fusion gives ~3.1x the TPR of `max()` at the same 1% FPR operating point, and a clear AUPRC
lead. Read plainly: 19.2% TPR at 1% FPR is a low absolute number. It should be, given what's
feeding it — L2 was trained for a single epoch on a corpus where 48% of attack rows carry no
family label (ADR-0005), L1's rule coverage is partial by the same investigation, and L3–L5
contribute nothing yet because they don't exist. This result is evidence for the cascade's
central premise — that combining several weak, cheap signals beats trusting the single strongest
one (ADR-0001) — not a claim that the detector is ready for the field. The project's real
headline number, TPR @ 0.1% FPR with a bootstrap CI over the full test set, stays `TBD` until
Milestone 9.

Full machine-readable output: [fusion-comparison.json](fusion-comparison.json).

## Policy engine

Five ordered verdicts (`ALLOW < FLAG < SANITISE < CHALLENGE < BLOCK`) over four configurable
thresholds, boundary-inclusive on the stricter side, with shadow mode as a first-class return
value rather than a separate code path (`decide(score, config, shadow=True)` still computes the
real verdict — only `enforced` changes). Design rationale: [ADR-0006](../adr/0006-knn-fusion-and-policy.md#decision-3--policy-five-ordered-verdicts-inclusive-upper-thresholds-shadow-mode-required).

No live-traffic numbers exist for policy yet — it has 8 passing unit tests (monotonicity across
all threshold boundaries, shadow-mode non-enforcement, inclusivity at exact threshold values) but
nothing calls it with real request volume until the gateway (Milestone 6) exists to generate any.

## What this milestone does and does not establish

Does: proves fusion beats a hand-tuned baseline on real held-out data, with real split discipline
and a real (if small-scale) measurement; ships a working, benchmarked kNN sidecar; ships a
policy engine with proven threshold and shadow-mode semantics.

Does not: wire any of this into a live request path (Milestone 6's job), establish the project's
actual headline metric (Milestone 9's job, at 0.1% FPR with a bootstrap CI over the full corpus,
not a 3,000-row sample at 1%), or make any claim about L3–L5's contribution, since they don't
exist yet.
