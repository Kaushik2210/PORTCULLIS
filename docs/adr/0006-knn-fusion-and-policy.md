# ADR-0007 — kNN sidecar design, fusion algorithm, and policy engine

**Status:** Accepted
**Date:** Milestone 5

## Context

Milestone 5 adds the kNN sidecar, combines it with L0/L1/L2 into a single calibrated score via
learned fusion, and adds the policy engine that turns that score into an enforcement decision.
Three decisions, each with a measured or reasoned basis rather than a default choice.

## Decision 1 — kNN sidecar: brute-force numpy, MiniLM-L6 embeddings, train-attack-only corpus

**Embedding model.** Reuses `sentence-transformers/all-MiniLM-L6-v2` — the same checkpoint as L2
(ADR-0005) — rather than a second, dedicated embedding model. It already is a general-purpose
sentence-embedding model; standing up a second model would cost a second download and a second
latency budget for no benefit this project needs yet.

**Index technology.** Plain numpy brute-force (`BruteForceIndex`), not FAISS. See
[ADR-0003's resolution](0003-faiss-over-hnswlib.md#resolution-at-milestone-5): measured on the
real corpus (6,831 attack rows, 384-dim), brute-force search is 0.69ms single-query, 0.26ms/query
batched. No ANN library earns its dependency cost at this scale.

**Corpus scope.** Attack rows (`label == 1`) from `train.jsonl` only — 6,831 rows. Benign rows
contribute nothing to "nearest known attack." Family labels come from `train.weak.jsonl`
(ADR-0005) where L1 could place one; rows L1 could not place still index with an empty family
string rather than being dropped, since a paraphrase of an unlabelled attack is still worth
retrieving as "resembles a known attack" even without a family name.

**A real inefficiency found and fixed while building this.** The first fusion-fitting run scored
rows through the kNN sidecar one text at a time and measured 2.1 rows/s — a ~30-minute run for a
Milestone-5 proof. The cause: embedding one text at a time through `sentence-transformers` pays a
fixed per-call overhead that batching amortises away entirely; the corpus-embedding step (6,831
rows in one batched call) had already measured ~42 rows/s, a fact the per-row scoring path simply
didn't use. Fixed by batch-embedding every row's query text up front
(`KnnScorer.score_batch`) before the per-row L0/L1/L2 loop. Re-measured: 141.6 rows/s on
validation, ~57 rows/s on the larger test sample (L2's ONNX cost dominates once embedding is no
longer the bottleneck) — roughly a 30-70x improvement, caught and fixed before spending the 30
minutes rather than after.

## Decision 2 — Fusion: logistic regression, not gradient-boosted trees

The spec named either as acceptable. Logistic regression was chosen for one reason that matters
more than raw accuracy at this stage: **its decision function is a sum of `weight * score` terms**,
which is exactly what the spec's dashboard Decision Inspector needs to show as "fusion
arithmetic" (`core/fusion/fuse.py` returns per-layer contributions alongside the final score,
directly from the coefficients). A gradient-boosted tree ensemble would need a separate
explanation method (SHAP values, feature importance) bolted on afterward to produce anything
similarly legible, and that gap between "what the model computed" and "what a human is shown" is
exactly the kind of thing NFR-5 (every decision returns a human-readable rationale) exists to
rule out. If GBT's accuracy advantage turns out to matter once Milestone 9's full harness runs,
this is revisitable — the interface (`LayerScores` in, `FusionResult` out) does not change if the
implementation behind `fuse()` does.

**Split discipline.** Fit on `validation` (790 rows), evaluated on a held-out sample of `test`
(deterministic stride of 3,000 rows from 19,937). Not fit on `train`: L2's classifier weights
were themselves fit on train (Milestone 4's 0.052 final loss means real memorisation of much of
it), so train rows are not a fair test of what fusion should learn to trust about L2's score.
Validation was previously used only for L2's *calibration* — two numbers, far lower capacity than
fusion's four coefficients — so it remains a fair, uncontaminated fitting set. The 3,000-row test
sample is a Milestone-5 proof point, not Milestone 9's rigorous, bootstrap-CI'd, full-19,937-row
headline number.

**Result — the checkpoint artifact this milestone exists to produce:**

| | TPR @ 1% FPR | AUPRC |
|---|---:|---:|
| Learned fusion | **0.192** | **0.640** |
| `max()` baseline | 0.062 | 0.545 |
| Delta | **+0.130** | **+0.095** |

Fusion clearly beats `max()` at this stage — roughly 3.1x the TPR at the same 1% FPR operating
point. Read honestly rather than triumphantly: 19.2% TPR at 1% FPR is a low absolute number, and
it should be — L2 was trained for one epoch on a corpus where over half the attack rows carry no
family label (ADR-0005), L1's 54 rules cover a fraction of real attack phrasing (ADR-0005's own
recall investigation), and L3-L5 do not exist yet to contribute anything at all. This number is
evidence that *combining* weak signals beats trusting the single strongest one — the whole
premise of the cascade (ADR-0001) — not evidence that the detector is production-ready. Fitted
weights: L2 dominates (6.42), L1 meaningfully positive (2.25), kNN meaningfully positive (1.58),
L0 near zero (0.009) - obfuscation alone is a weak standalone signal, consistent with ADR-0001's
framing of L0 as a feature that matters combined with content, not in isolation.

## Decision 3 — Policy: five ordered verdicts, inclusive-upper thresholds, shadow mode required

`ALLOW < FLAG < SANITISE < CHALLENGE < BLOCK` over four configurable cut points
(`PolicyConfig`), per tenant/route as the spec requires (a `PolicyConfig` instance is the unit
that varies per tenant, not a global). A score exactly on a threshold gets the *stricter*
verdict, not the more lenient one — a fixed-FPR operating point is meaningless if scores landing
precisely on the boundary fall through to the softer action by an off-by-one.

Shadow mode (`decide(..., shadow=True)`) still computes the real verdict and rationale, only
`enforced` differs. The spec calls shadow mode out as how a detector like this actually gets
adopted into a real deployment - log every request's would-be verdict, build trust in the
false-positive rate, then flip enforcement on - and this ADR treats that as load-bearing enough
to be the default calling convention (`decide()` returns a full result either way) rather than a
separate code path that could silently drift from the enforced one.

## What Milestone 5 does not include

- Live gateway wiring. `training/fusion/fit.py` calls L0->L1->L2->kNN in sequence for
  fitting/evaluation; nothing yet calls that sequence per live request. Milestone 6 (FastAPI
  proxy) is where that wiring becomes necessary and gets built.
- L3-L5 as fusion inputs. `LayerScores` has four fields because four layers exist. Adding a
  fifth is a field addition and a refit, not an interface change.
- The real headline metric. TPR @ 0.1% FPR on the full test set with a bootstrap CI is
  Milestone 9's job. This ADR's table is a proof that fusion beats max(), on a smaller sample, at
  a coarser (1%) FPR this sample can actually support.
