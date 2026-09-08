# ADR-0010 — Milestone 9 eval harness: scope, methodology, and cuts

**Status:** Accepted
**Date:** Milestone 9

## Context

M9 is the spec's largest single milestone: a headline benchmark table, baselines, ablations, and
adaptive-attacker evaluation. Several of its named pieces need a real external cost or dependency
commitment (an LLM API, a commercial detection API, three red-team tools wired into scheduled CI,
a load-testing tool) that shouldn't be decided silently. Raised with the user directly before any
code; agreed scope below.

## Decision 1 — scope: core measurement only, expensive tail cut and documented

**Built:** the headline table (TPR @ 0.1%/1% FPR with bootstrap CIs, AUPRC, per-layer
contribution), ablations (remove L0, remove kNN, `max()` vs. learned fusion), a regex-only
baseline, full-cascade latency percentiles, and two adaptive-attacker techniques that need no new
dependency: rule-based paraphrase attacks and iterative black-box query attacks.

**Cut, each for a stated reason, not silently dropped:**
- **LLM-as-judge and commercial-API baselines** - both need a live API key and real per-run
  spend. Not authorised.
- **Off-the-shelf `deberta-v3-base-injection`** - a new model download; deferred alongside the
  other baselines rather than singled out, since the value of a baseline table is comparing
  against *all* of them together, not one in isolation.
- **garak/PyRIT/promptfoo wired into scheduled CI** - three heavy new tools plus recurring GitHub
  Actions spend for a portfolio project with no production traffic to protect yet.
- **k6/Locust load testing** - a new external tool, and meaningfully different work (standing up
  the gateway as a live load target, not scoring a static test set).
- **GCG-style gradient-based transfer-suffix attacks** - genuinely research-grade: white-box
  gradient access to L2's model, typically hours of search per successful suffix even on a GPU.
  This machine is CPU-only (ADR-0002). Sinking hours into a technique this project's own hardware
  can't run well would produce a weak, unrepresentative result - worse than not reporting one.

## Decision 2 — per-attack-family recall is not reported on the test set, and why

The spec's headline table asks for per-family recall. The test corpus carries only a binary
label plus provenance fields (`source`, `license`, `tier`) - no family/category ground truth.
The only family labels anywhere in this project come from L1's own weak-labeling
(`training/l2/weak_label_corpus.py`), and that module's docstring already states, since
Milestone 4, exactly why it was never run on `test.jsonl`: *"to keep L1's own eval (Milestone 9
ablations) independent of what trained L2."* Weak-labelling the test set now, just to fill in a
per-family column, would mean L1's own rule firing supplies both the family ground truth *and*
half of what's being measured against it - directly circular for the regex-only baseline and for
any ablation involving L1.

**Decision: per-family recall is a documented omission, not a fabricated column.** The project's
per-family *behaviour* is still visible elsewhere, honestly: ADR-0005's weak-label coverage story
(which families L1 can and cannot name), M7's real finding (repeated override attempts pushing
conversation state to PROBING while each individual turn stays under threshold), and this
milestone's own adaptive-attack results (which families of evasion actually work against the
real system). A single fabricated recall-by-family table would look more complete while being
less honest than what's already published.

## Decision 3 — bootstrap CI methodology

`metrics.py`'s `bootstrap_tpr_at_fpr` resamples the test set's `(label, score)` pairs **with
replacement**, `n_resamples` times (2000 by default - enough for a stable 95% CI without an
unreasonable runtime), recomputing TPR at the nearest FPR-at-or-below the target on each
resample, and reports the 2.5th/97.5th percentiles as the CI. Scoring the ~20k test rows through
the real cascade happens **once**; the bootstrap itself resamples the already-computed
`(label, score)` array, not the model - this is what keeps 2000 resamples computationally cheap
(pure numpy/sklearn operations) rather than requiring 2000 reruns of L2 inference.

## Decision 4 — ablation methodology

Each ablation refits fusion's logistic regression on `validation` (M5's own fit set, unchanged)
with a feature held out or replaced, then evaluates the refit model on the same scored `test`
set the headline number uses - not a separate, smaller sample. "Remove L0" and "remove kNN" mean
excluding that feature from the regression entirely (a 3-feature fit), not zeroing the input at
inference time, so the remaining features' coefficients are refit to compensate, the fair
question being "how much worse is the *best possible* fusion without this signal," not "how much
worse is today's fusion if you break one input."

**L4 has no ablation entry.** It operates on conversation history, not a single test-set row -
there is no per-row "with/without L4" comparison to run. Its measured contribution is M7's own
report (a real conversation escalating state while every individual turn stays under the flag
threshold), referenced here rather than forced into a table shape it doesn't fit.

## Decision 5 — adaptive attacks: two techniques, no new dependency

Both assume the attacker can query the deployed system (grey/black-box, per the spec's threat
model - ADR-0001's adversary (b) and (c)), and both operate as a search over text mutations
scored by the same fused score a real request would get:

- **Paraphrase attacks** (`adaptive/paraphrase.py`) - a small hand-authored synonym/phrasing
  table for the vocabulary that shows up in known attack strings (`ignore`->`disregard`/`forget`,
  `instructions`->`directions`/`rules`, `reveal`->`show`/`expose`, etc.), applied combinatorially
  to seed attack strings drawn from the real test set's positive rows. Deliberately not
  LLM-generated - the point is measuring whether *cheap, mechanical* rephrasing evades detection,
  which is the realistic low-effort end of what an actual attacker tries first.
- **Iterative black-box query attacks** (`adaptive/blackbox_query.py`) - greedy hill-climbing:
  starting from a real seed attack string, repeatedly try a pool of candidate mutations
  (character substitution via L0's own confusables table, word deletion, whitespace injection,
  case changes), keep whichever mutation reduces the fused score the most, stop once the score
  crosses below the block threshold or a query budget is exhausted. A simplified relative of
  established black-box adversarial-text search methods (e.g. TextFooler's greedy word-importance
  search), not a re-implementation of any of them.

Both report **attack success rate honestly, including where it's bad** (spec's own framing) -
this is explicitly not a table meant to end at a reassuring number.

## What Milestone 9 does not establish

- Everything named in Decision 1's cut list.
- A load-tested throughput figure at the latency SLO - the spec's own ask, cut per Decision 1.
- Any claim that the adaptive attacks here are a complete adversarial evaluation - they are two
  concrete, reproducible, dependency-free techniques, not an exhaustive red-team exercise. A
  real adversarial evaluation additionally needs the tools cut in Decision 1.
