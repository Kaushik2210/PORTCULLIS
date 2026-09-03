# ADR-0001 — Layered detection cascade

**Status:** Accepted
**Date:** Milestone 0
**Supersedes:** none

## Context

PORTCULLIS sits in the request path of an LLM application. It must decide, per request, whether
the input subverts the instruction hierarchy. Three forces conflict:

1. **Latency.** The gateway is synchronous. Every millisecond is added to the user's
   time-to-first-token, on top of the model's own latency.
2. **Explainability.** A block with no explanation is unactionable. Security teams will not
   deploy an enforcing control they cannot audit, and "the model said 0.87" is not an audit
   trail. This is a hard requirement, not a nice-to-have.
3. **Coverage.** Attacks range from copy-pasted jailbreaks to novel paraphrases to multi-turn
   crescendos to payloads that arrive inside a retrieved PDF. No single mechanism covers that
   range.

## Decision

Detection is a **cost-ordered cascade of heterogeneous layers with short-circuit evaluation**,
not a single classifier and not an LLM judge.

```
L0  normalisation & de-obfuscation   (always runs, ~microseconds)
L1  signature & heuristic engine     (always runs, fully explainable)
L2  learned classifier + kNN sidecar (runs when L1 is not decisive)
L3  LLM adjudication                 (uncertainty band only, target <5% of traffic)
L4  conversation state machine       (cross-turn, Redis-backed)
L5  egress inspection                (model output, incl. canaries)
     |
     +-- learned fusion over layer scores --> calibrated probability --> policy
```

Layers are **heterogeneous by design**: surface-form rules, learned semantics, embedding
geometry, and temporal behaviour fail in different ways. That diversity is the point (threat
model, adversary C) — an evasion must defeat all of them simultaneously.

Cheap layers short-circuit. A request that L1 matches with high confidence never reaches the
classifier; a request the classifier is confident about never reaches L3.

### Load-bearing consequences

- **L0 runs before everything, unconditionally.** A classifier trained on clean text is trivially
  bypassed by base64 or homoglyphs. Normalising after scoring would be useless, so L0 is not
  optional and not a layer that can short-circuit.
- **Obfuscation is scored, not just removed.** L0 emits a `normalization_report`; decode depth
  and transform count are features. Legitimate prompts rarely contain triple-nested base64.
- **Untrusted-provenance content runs the same cascade at a stricter threshold.** Retrieved
  documents and tool results are not user turns and must not share their threshold.
- **Every layer returns spans, not just scores.** The explainability requirement is satisfied
  per-layer or not at all; a fused score with no matched span cannot be explained after the fact.

## Alternatives rejected

**Single fine-tuned classifier.** Simplest to build and the most common approach. Rejected on
explainability: it cannot say *why*, only *how much*. It also has one failure mode, which
adversary C only has to find once. Retained as an internal layer (L2), not as the architecture.

**LLM-as-judge for everything.** Best raw coverage and near-zero engineering. Rejected on three
counts: latency (a model call per request, doubling TTFT), cost (per-request inference on 100%
of traffic), and — decisively — **the judge is itself injectable**. Retained as L3 for the
ambiguity band only, where its cost is bounded and measured.

**Regex/signatures only.** Fast, perfectly explainable, zero infrastructure. Rejected on
coverage: brittle under paraphrase by construction. Retained as L1, where being fast and
explainable is exactly what is wanted, and where its brittleness is backstopped.

**Parallel evaluation of all layers, then fuse.** Simpler control flow and lower tail latency
variance. Rejected because it forfeits the entire cost argument: every request would pay for L2
and L3. Reconsider if measurement shows short-circuit rates are low enough that the branch
prediction is not worth the complexity — this is an empirical question the eval harness answers.

## Consequences

**Positive.** Cost and latency scale with ambiguity rather than volume. Each layer is testable in
isolation. Ablations are natural, so the architecture can be *proven* non-arbitrary rather than
asserted. Explanations compose from per-layer spans.

**Negative.** More moving parts than a single model. Fusion is an extra component that must be
trained and can itself be wrong. Short-circuit thresholds are a tuning surface, and a badly set
L1 threshold silently starves L2 of traffic — the eval harness must report per-layer traffic
share, not only per-layer accuracy.

**Neutral.** Layer count is not fixed. Layers may be added or dropped if ablation shows a layer
does not pay for itself. That test is a requirement of Milestone 9, and a layer that fails it
gets removed rather than defended.
