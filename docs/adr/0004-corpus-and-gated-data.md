# ADR-0004 — Corpus backbone and the gated-data boundary

**Status:** Accepted
**Date:** Milestone 0

## Context

Two project non-negotiables collide.

- **NFR-6:** everything is reproducible — pinned deps, seeded training, dataset manifests with
  content hashes, one-command rebuild from scratch.
- **The data plan** named LMSYS-Chat-1M as the benign backbone and
  `deepset/prompt-injections` + `jackhhao/jailbreak-classification` as the adversarial core.

Verification at Milestone 0 found both halves of that plan unworkable as stated.

### Finding 1 — the named adversarial sets are far too small

| Dataset | Rows | Licence | Note |
|---|---|---|---|
| `deepset/prompt-injections` | 662 | Apache-2.0 | Mixed German/English |
| `jackhhao/jailbreak-classification` | ~1,306 | Apache-2.0 | |

Roughly 2k rows combined. That is not enough to fine-tune a 149M-parameter encoder, and it is
nowhere near enough for the headline metric — see Finding 3.

### Finding 2 — the benign backbone is gated in a way that breaks reproducibility

LMSYS-Chat-1M requires a click-through licence agreement with name, email, affiliation and
country, and the agreement includes a **right to require deletion of all copies at any time**.
A dataset that can be withdrawn cannot sit inside a one-command rebuild, and a manifest hash
does not help when the source is gone.

### Finding 3 — the headline metric constrains the test set size

TPR @ 0.1% FPR means the operating threshold is set where 1 benign example in 1,000 is
misclassified. Estimating that rate requires enough benign test examples for the count of false
positives to be a stable number. With a ~2k-row corpus the benign test split yields single-digit
false positives, and the resulting TPR has confidence intervals wide enough to make the point
estimate meaningless. Publishing it unqualified would be the kind of number a reviewer checks
and discards the project over.

## Decision

**1. Corpus backbone becomes `hendzh/PromptShield`** — 43,425 rows, Apache-2.0, ungated,
train/validation/test splits provided. The previously named sets are retained as *supplementary
and cross-source evaluation slices*, which is where small, independently-collected sets are
genuinely valuable: measuring transfer to a source the model did not train on.

**2. A hard tier boundary between ungated and gated data.**

- *Tier 1 (default)* — ungated, redistributable-manifest sources only. `just eval` rebuilds
  this tier end-to-end on a clean machine with no accounts and no manual steps.
- *Tier 2 (opt-in)* — gated sources, behind an explicit `--include-gated` flag, skipped by
  default with a clear log line rather than a crash.
- **Every published number records which tier produced it.** The dataset manifest carries the
  tier, source, licence and content hash per row group.

**3. The headline metric ships with bootstrap confidence intervals.** TPR @ 0.1% FPR is reported
as a point estimate *and* a 95% bootstrap CI over the test set. If the CI is too wide to support
a claim, the README says so. The harness additionally reports the benign test-set size next to
the metric, so a reader can judge the estimate without trusting the CI computation.

## Alternatives rejected

**Keep LMSYS as the backbone and document the manual step.** Rejected: it converts
"one-command rebuild" into "one command, plus an account, plus an approval wait, plus a licence
that may be revoked". The reproducibility claim would be false in the only case that matters —
someone else trying to reproduce it.

**Synthesise the benign corpus with an LLM.** Cheap, ungated, unlimited. Rejected as the primary
source: synthetic benign text has a different distribution from real user traffic, and the whole
FPR-first thesis depends on the benign distribution being realistic. A detector tuned against
synthetic negatives will have a beautiful reported FPR and a terrible real one. Retained only for
targeted augmentation of the hard-negatives corpus, train split only.

**Report accuracy on a balanced set instead.** Rejected — it is the vanity metric this project
exists to argue against, and it is explicitly barred from the README.

## Consequences

- Tier 1 is smaller than the theoretical maximum corpus. Accepted: a reproducible number beats a
  larger unreproducible one.
- Two eval paths must be maintained, and CI only ever exercises Tier 1.
- The hard-negatives corpus, built by hand, remains the highest-value asset in the repo and is
  unaffected by this decision — it is Tier 1 by construction, since we author and licence it.
- If the Tier 1 benign test split still cannot support a 0.1% FPR estimate once assembled, the
  honest response is to report at **1% FPR** as the headline and 0.1% as secondary with its CI —
  *not* to publish a number the data cannot support. This is a decision to revisit at
  Milestone 3 with the split statistics in hand.
