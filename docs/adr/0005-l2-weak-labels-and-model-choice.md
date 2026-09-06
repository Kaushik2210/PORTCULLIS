# ADR-0005 — L2 label source, model choice, and training scope

**Status:** Accepted
**Date:** Milestone 4

## Context

L2 is specified as an 8-class multi-label classifier (7 attack families + benign). The Tier-1
corpus built at Milestone 3 carries binary ground truth only — PromptShield and its supplements
say *attack or not*, never *which kind*. The dataset card flagged this as a known gap with two
possible fixes: weak-label via the L1 rule taxonomy, or find/build a labelled multi-class source.
No labelled multi-class source of comparable size and licence exists. This ADR picks weak
labelling and records what that choice actually costs, discovered by measurement, not assumed.

Separately: the original spec called for fine-tuning and comparing three candidate encoders to
build a real latency/TPR frontier before choosing one. There is no GPU on this machine (M0).
The user chose, in a scope discussion at the start of Milestone 4, to train only the M0 latency
leader (MiniLM-L6) end-to-end and defer the full three-model comparison rather than spend a
multi-hour CPU commitment this session cannot back with a GPU.

## Decision

### Label source: weak supervision via L1, binary-first

An attack row's binary label is trusted — it came from the source corpus. Its *family* is
assigned by running the row through the full L1 rule corpus (`portcullis.training.l2.labels`):
every taxonomy label any rule matched becomes a multi-hot bit. A benign row needs no corroboration
and is always labelled (only the `benign` bit set). **An attack row that matches no L1 rule is
excluded from multi-label training, not guessed at.** Guessing a family for it would be
indistinguishable from noise; leaving it unlabelled and out of the loss is honest about not
knowing.

This makes L2's multi-label training signal **a function of L1's own coverage.** A genuinely
novel attack shape L1's rules do not recognise cannot be weak-labelled, so L2 cannot be shown it
carries a family. **L2 trained this way is not independently verified to generalise to attack
families L1 cannot already name** — that would require ground truth from a source other than L1,
which does not exist yet. This is the real cost of this label source, stated plainly rather than
elided. The binary detection task (attack vs. benign, the metric the whole eval harness is built
around) does not have this problem: it comes straight from the trustworthy source-corpus label,
never from L1.

### A measured finding, not a footnote: L1's rules were overfit to their own examples

The first coverage run excluded **94.1%** of train attack rows — only 5.9% of confirmed attacks
matched any of the 50 rules shipped at Milestone 2. That number was investigated before being
accepted, the same way the M0 latency anomaly and the M3 dedup chaining bug were: by reading
actual excluded rows rather than trusting the aggregate.

Three specific, generalisable gaps explained most of it, all present because M2's rules were
authored and tested only against examples I wrote myself:

1. **Word order.** `PC-OVR-001` required the qualifier before the noun ("ignore *previous*
   instructions"). Real text just as often puts it after ("ignore all instructions and rules
   *above*"). Same issue in `PC-OVR-004`'s "new instructions supersede X" — "from now on, *new
   instructions*" puts the cue first.
2. **Passive voice.** "All previous instructions *should be abandoned*" shares no verb with
   `PC-OVR-001`'s active-voice list (`ignore`, `disregard`, ...). Added `PC-OVR-009` rather than
   overloading the existing rule.
3. **A missing pattern entirely.** A "remember this secret key: `XXXXXX`, don't ever tell it to
   anyone" template — the single largest near-duplicate cluster found during Milestone 3 dedup,
   500+ rows — had no rule at all. Added `PC-EXF-008`.
4. **A missing adverb form.** "Disregard all *previously provided* instructions" — `previous`
   (adjective) was covered, `previously` (adverb) was not. One-token fix to `PC-OVR-001`.

Each fix was re-verified against the full L1 test suite (184 tests, including the hard-negative
and cross-contamination checks) before being trusted, and each was re-measured against the real
corpus, not assumed to help:

| Stage | Train exclusion rate |
|---|---:|
| Original 50-rule corpus | 94.1% |
| + word-order fixes, passive voice, secret-key rule | 54.4% |
| + adverb-form fix | **48.0%** |

**Stopping point.** 48% residual exclusion is still substantial, and it is left as a measured,
documented gap rather than chased further this milestone. A full per-attack-family recall audit
is explicitly Milestone 9's job (the spec's own ablation table). Continuing to patch rules against
one held-out corpus without a principled stopping rule is how a signature set drifts into
overfitting the very data it will be measured against — the discipline this project applies
against L2 overfitting its training set applies equally to L1 overfitting its rule-tuning corpus.

### Model: MiniLM-L6 only, three-model frontier deferred

`sentence-transformers/all-MiniLM-L6-v2` (22M params) — the M0 latency leader (ADR-0002) at every
sequence length measured, and the smallest of the three surviving candidates after ModernBERT-base
was eliminated. Trained here to prove the full pipeline (weak-label → train → calibrate → export
→ measure) end-to-end with real numbers. DistilRoBERTa-base and DeBERTa-v3-small are not trained
in this milestone; the latency/TPR frontier the spec asks for remains a real gap, recorded here
rather than silently dropped, and is the natural target of a future session with either more CPU
time budgeted or GPU access.

## Consequences

- L2's multi-label predictions inherit L1's blind spots by construction. Any claim that L2 "adds
  coverage L1 lacks" needs evidence from Milestone 9's ablation, not from this training run alone.
- The binary head (the one the headline TPR@FPR metric depends on) is unaffected by weak-label
  noise — it trains on the trustworthy source label directly.
- Rows excluded from multi-label supervision are **not** excluded from the corpus generally, only
  from the per-row multi-label loss; they remain in the corpus for the binary task.
- The L1 rule improvements made here are a real, if incidental, Milestone-2 quality improvement,
  not just an M4 side effect — the fixed rules stay in `rules/` regardless of what happens to L2.
- Reopening this ADR is warranted if Milestone 9's per-family recall numbers show the residual 48%
  exclusion is concentrated in one or two attack families rather than spread evenly, since that
  would point at another specific, fixable rule gap rather than genuine novelty L1 cannot express.
