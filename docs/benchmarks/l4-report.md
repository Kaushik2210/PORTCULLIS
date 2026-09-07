# L4 conversation state machine — Milestone 7

What got built, what the real checkpoint actually does with it, and an honest account of a
finding that shaped the demo. Design rationale (states, signals, the policy-modifier decision,
storage) is in [ADR-0008](../adr/0008-l4-conversation-state-machine.md); this report is the
measurement side of that ADR - and, per that ADR's own Decision 3, there is no benchmark number
here, because no multi-turn labeled corpus exists to produce one against.

## What was built

- `packages/core/src/portcullis/core/l4/` - the state machine itself. `advance()` is a pure
  function (`ConversationRecord`, `TurnSignal`, `now`) -> `ConversationRecord`; `apply_floor()`
  turns the resulting `ConversationState` into a verdict floor. `ConversationStore` is a
  `Protocol`; `InMemoryConversationStore` (the one concrete backend that belongs in core) is
  used by every test and is the gateway's default when no Redis is configured.
- `packages/gateway/src/portcullis/gateway/conversation.py` - `ConversationAwareDetector` wraps
  a `DetectionPipeline` (M6) with L4: looks up prior state, embeds the turn, advances, applies
  the floor, persists. Opt-in via `conversation_id`; omitted, it's exactly M6's behaviour.
- `packages/gateway/src/portcullis/gateway/redis_store.py` - `RedisConversationStore`, the
  production backend, tested against the real `ops/compose.yaml` Redis service (`needs_infra`).
- `conversation_id` added to `/v1/detect`, `/v1/detect/batch` and `/v1/chat/completions` (an
  additive PORTCULLIS extension, stripped before anything is forwarded upstream).

## Testing

26 pure unit tests for the state machine and its store (`packages/core/tests/test_l4_machine.py`,
`test_l4_store.py`) - the "write the failing test first" detection-logic tests the working
agreement asks for, and they earned their keep: two real calibration bugs were caught by tests I
wrote against my own implementation, before either shipped:

1. **Decay formula was wrong.** `DEFAULT_HALF_LIFE_S` was named and documented as a true
   half-life ("decays by half every 5 minutes"), but the implementation used
   `exp(-elapsed/half_life)` - an e-folding time, which decays to ~37% after that duration, not
   50%. `test_decay_over_one_half_life_roughly_halves_prior_risk` failed immediately. Fixed to
   `0.5 ** (elapsed / half_life)`.
2. **Benign chatter geometrically accumulated risk.** With no floor, repeated near-zero scores
   (ordinary benign text scores ~0.02-0.03, per M6's own demo probe) still summed under a slow
   decay and frequent turns - the geometric series converges to `score / (1 - decay)`, which for
   a 10-second turn cadence and a 5-minute half-life is nowhere near zero. Ten purely benign
   turns pushed cumulative risk to 0.18, past the PROBING bar, on volume alone.
   `test_repeated_benign_turns_never_leave_normal` failed. Fixed by adding a noise floor: a turn
   only contributes `max(0, score - 0.05)`, not its raw score - purely benign chatter now
   contributes exactly zero, forever, regardless of frequency.

A third test (`test_a_single_spike_after_calm_turns_is_not_a_crescendo`) caught a related design
flaw: "non-decreasing" run detection let a flat calm sequence plus one later spike count as a
single sustained climb back to the start. Fixed by requiring *strictly increasing* consecutive
turns for a crescendo, not merely non-decreasing.

Gateway-level tests follow the same `needs_model`/fake-pipeline split M6 established:
fake-pipeline tests (`test_app.py`) prove the wiring - conversation state escalates a verdict
fusion alone would have allowed, state doesn't leak across `conversation_id`s,
`conversation_id` is stripped before proxying - using a deterministic fake embedder, no real
model. Real-pipeline tests (`test_conversation_real.py`, `needs_model`) prove the same shape of
claim against the actual trained checkpoint and real sentence-transformer embeddings.

## What the real checkpoint actually shows

Four varied real phrasings of the same override attempt, sent through the real pipeline as one
conversation (`just demo-m7`, Part 1):

| Turn | Score | Fusion verdict alone | Conversation state |
|---|---:|---|---|
| 1 | 0.1805 | ALLOW | NORMAL |
| 2 | 0.1209 | ALLOW | PROBING |
| 3 | 0.1336 | ALLOW | PROBING |
| 4 | 0.1280 | ALLOW | PROBING |

**Every single turn scores under the default flag threshold (0.2) on its own - fusion alone would
allow all four.** Conversation state moves to PROBING by turn 2 regardless, because cumulative
risk (not any one turn) crossed the bar. This is L4's entire premise, working end to end on real
model output: no individual message was severe enough to act on, but the pattern of repeated
boundary-testing was visible to the layer built to look for exactly that.

**A finding worth being direct about: this is cumulative-risk accumulation, not crescendo.**
The four scores above are not strictly increasing (0.18 -> 0.12 -> 0.13 -> 0.13) - real,
noisy model output, not a clean ramp. Crescendo detection (a strictly-increasing run) never
fires on this real sequence. An earlier probe with more gradually-escalating *phrasing*
("questions about safety filters" -> "pretend you have no filters" -> the direct override) also
failed to produce a rising score sequence - the real checkpoint scored the first three turns
almost identically (~0.024-0.026), distinguishing sharply only on the blatant final phrase. This
is consistent with what M4/M5 already documented about this model (one-epoch training, narrow L1
coverage): it is not sensitive to gradual escalation in phrasing, only to fairly direct language.

Rather than search for a real prompt sequence that happens to produce a rising score curve just
to make the milestone's named checkpoint ("crescendo caught") look cleaner, the demo says this
plainly and shows the crescendo *algorithm* separately, against a constructed synthetic score
sequence (`just demo-m7`, Part 2, no model involved) - proving the algorithm itself is correct
(also covered by `test_l4_machine.py`), while being honest that a live crescendo through the
current real classifier is not something this milestone produced evidence for.

## What Milestone 7 does not establish

- Any benchmarked detection quality for L4 - by design, per ADR-0008 Decision 3. No multi-turn
  labeled corpus exists; the numbers above are one real, reproducible measurement, not a TPR/FPR
  claim.
- A demonstrated real crescendo (strictly-increasing scores) against the trained classifier -
  the section above says so directly rather than curating a prompt sequence to manufacture one.
- Role-drift or topic-pivot-after-refusal against real model output - both are exhaustively unit
  tested in core (synthetic signals) and exercised in the gateway's fake-pipeline suite, but
  Part 1's real scenario only exercises cumulative-risk accumulation; constructing real text that
  reliably triggers the other two signals through the actual checkpoint is undone work.
- Fusion retraining or any change to M5's weights - L4 is a policy modifier, not a fusion input,
  precisely so this milestone would not need data it doesn't have (ADR-0008 Decision 1).
