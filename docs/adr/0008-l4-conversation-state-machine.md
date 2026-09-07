# ADR-0008 — L4 conversation state machine

**Status:** Accepted
**Date:** Milestone 7

## Context

Milestone 7 adds L4: the layer that catches what stateless detection structurally cannot -
attacks assembled across turns rather than sitting inside any single one. The spec names four
signals (cumulative risk with time decay, crescendo detection, role-drift, topic-pivot-after-
refusal) and a four-state machine (`NORMAL -> PROBING -> ESTABLISHING -> EXPLOITING`), and asks
for a Redis-backed store. None of the shape of *how* those signals combine into transitions, or
how the resulting state should affect a verdict, was specified precisely enough to implement
without deciding it first - so, as with M0's threat model and M6's scope questions, the design
was worked out and checked against the user before code.

## Decision 1 — policy modifier, not a fifth fusion input

Raised directly: should L4's signal join `LayerScores` as a fifth fusion feature (refit alongside
l0/l1/l2/knn), or sit outside fusion as a policy-level modifier?

**Decision: policy modifier.** Fusion's weights (M5) were fit against ~19k independent,
single-turn corpus rows. There is no multi-turn labeled corpus to fit an `l4` coefficient
against - PromptShield and this project's own hard-negatives are single-turn by construction
(ADR-0004). Adding `l4=0` to every training row and refitting would not calibrate anything; it
would just teach the regression that the feature is uninformative, which is an artifact of the
training data's shape, not a real statement about conversation state's predictive value.

Instead, L4 computes a `ConversationState` and maps it to a **verdict floor** - `advance()`
returns a state, and a fixed table (not fitted, explicit and tested for monotonicity like
`PolicyConfig`'s thresholds) says the minimum verdict a request in that state receives,
regardless of what fusion alone would have said:

| State | Verdict floor |
|---|---|
| `NORMAL` | none (fusion's own verdict stands) |
| `PROBING` | none (still gathering evidence - a floor here would flag ordinary conversations) |
| `ESTABLISHING` | `FLAG` |
| `EXPLOITING` | `CHALLENGE` |

The final verdict is `max(fusion_verdict, state_floor)` (`Verdict` is already ordered, M5). This
is a rule, not a fitted weight - it can be argued about and changed without needing data that
doesn't exist, and the decision trace can say plainly "conversation state ESTABLISHING raised the
floor to FLAG" rather than attributing the escalation to an uninterpretable regression
coefficient.

## Decision 2 — the four signals, concretely

`packages/core/src/portcullis/core/l4/machine.py`'s `advance()` is a pure function:
`(ConversationRecord, TurnSignal, now: float) -> ConversationRecord`. No wall-clock reads inside
it - `now` is passed in, so the state machine is exactly as testable as L0-L2 and fusion already
are (deterministic given its inputs, no hidden time dependency).

- **Cumulative risk with time decay.** `risk = risk * exp(-elapsed_s / half_life_s) + turn.score`
  on every turn, `elapsed_s` measured against the *previous* turn, not the conversation start -
  a burst of turns decays slowly between them; a conversation resumed after a long gap decays
  most of its accumulated risk, matching the spec's framing of decay as "was this ramping up
  quickly, or is this an old, cooled-off thread." `half_life_s` defaults to 300s (5 minutes) -
  disclosed as a reasoned default, not a fitted value (see Decision 3).
- **Crescendo detection.** The bounded turn history (last 8 turns) is checked for a run of 3+
  consecutive turns with non-decreasing scores and a total rise of at least 0.15 - "monotonic
  escalation," not merely "the last turn was worse than average." A single spike is not a
  crescendo; a sustained climb is.
- **Role-drift.** If any earlier turn's taxonomy labels (from L1, already computed per-turn by
  M6's pipeline) included `role_play_jailbreak` - a persona-setting move - and a *later* turn
  fires a different, higher-severity label (`direct_override`, `system_prompt_extraction`,
  `data_exfiltration`, `tool_abuse`), that is exactly the spec's "persona established turn 3,
  exploited turn 9" shape.
- **Topic-pivot-after-refusal.** If the previous turn's verdict was `CHALLENGE` or `BLOCK` (a
  refusal-worthy turn) and the current turn's text is semantically dissimilar to it (cosine
  similarity below 0.3), that is a topic pivot immediately after being pushed back on - the shape
  of someone routing around a refusal rather than continuing the same request. Both the embedding
  and the similarity are computed by the caller (the gateway already has the sentence-transformer
  loaded for the kNN sidecar, M5, and its vectors are already L2-normalised, so cosine similarity
  is a plain dot product); `core` receives only the resulting float
  (`TurnSignal.topic_similarity_to_previous`) and compares it against the threshold - it never
  imports a sentence-transformer or does vector math, so `portcullis-core` gains no new dependency
  here. `ConversationRecord.last_embedding` carries the previous turn's vector opaquely between
  calls (`advance()` stores whatever it's given and never reads the values inside) purely so the
  *next* `advance()` call has something to derive a similarity from without a second store lookup.

State transitions combine these: `PROBING` on the first sign of risk (decayed cumulative risk
crosses a low bar, or any single turn scores above the flag threshold without crossing higher);
`ESTABLISHING` on a persona-setting label or a detected crescendo while in `PROBING`;
`EXPLOITING` on role-drift, a topic pivot after refusal, or decayed cumulative risk crossing a
high bar while already in `ESTABLISHING`. Full transition table and the tests proving it in
`packages/core/tests/test_l4_machine.py`.

## Decision 3 — an honest limitation, stated up front

**There is no multi-turn labeled corpus to measure any of this against.** Milestone 3's data
pipeline built a family/source-disjoint split of single-turn rows; nothing in this project's data
plan produces labeled multi-turn conversations. The thresholds above (half-life, crescendo rise,
similarity cutoff, risk bars) are reasoned defaults, not fitted or benchmarked values - the same
honest category as M6's placeholder policy thresholds (`gateway-report.md`).

This milestone's checkpoint artifact is explicitly "multi-turn crescendo caught in a recorded
demo" (spec's milestone table) - a scripted, illustrative conversation, not a measured TPR/FPR
claim. `docs/benchmarks/l4-report.md` says this plainly rather than letting a demo that works
imply a capability that has been benchmarked. A real evaluation needs either a labeled multi-turn
corpus (none exists) or an adversarial multi-turn generator - `garak`/`PyRIT` (spec's external
red-team CI section) are the closest fit and are Milestone 9's job, not this one's.

## Decision 4 — storage: a Protocol in core, two implementations elsewhere

`core/l4/store.py` defines `ConversationStore` as a `Protocol` (`get`, `set`, both keyed by
conversation id, `set` taking a TTL) - no concrete backend, so `portcullis-core` stays
dependency-light (ADR-0001's "the detection path must never import torch" extends here: it must
never import an infra client either). Two implementations:

- **`InMemoryConversationStore`** lives in `core` itself - it's pure Python (a dict with
  manual TTL eviction), no infra dependency, and every L4 unit test and the gateway's fake-
  pipeline test suite use it, so `just test` never touches Docker.
- **`RedisConversationStore`** lives in `packages/gateway` (`redis_store.py`), using
  `redis.asyncio`. `redis` was not named as a specific *client library* in the spec's stack list
  (only "Redis" the service was), the same category of gap `httpx`/`uvicorn` were at M6 - treated
  as implied by the already-approved Redis entry rather than re-asked, and disclosed here rather
  than added quietly. Tested separately, marked `needs_infra` (existing project convention),
  against the real `ops/compose.yaml` Redis service.

`ops/compose.yaml`'s Redis was already scaffolded at Milestone 0 with exactly the config L4
needs (`maxmemory-policy allkeys-lru`, no persistence, TTL-friendly) - a case of an earlier
milestone's forethought paying off rather than needing rework now.

## What Milestone 7 does not include

- Any real measurement of L4's detection quality - Decision 3 states why, plainly.
- Fusion retraining - Decision 1.
- `conversation_id` is opt-in on both `/v1/detect` and `/v1/chat/completions` (a new optional
  request field, defaulting to absent). Omitting it reproduces exactly M6's stateless behaviour -
  existing M6 tests and callers are unaffected. A dashboard or SDK that wants conversation
  tracking supplies its own id; this milestone does not add session/cookie machinery to invent
  one automatically.
- Rate limiting - named alongside conversation state in the spec's Redis bullet, but a distinct
  concern with its own design questions, left for whenever it gets its own milestone attention.
