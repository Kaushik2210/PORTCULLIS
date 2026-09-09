# ADR-0011 — Dashboard: data sources, and reusing M9's ablation as the model diff

**Status:** Accepted
**Date:** Milestone 10

## Context

The spec asks for three dashboard surfaces (live traffic, decision inspector, red-team console)
backed by a Next.js 15 App Router / TypeScript strict / Tailwind v4 / shadcn/ui / TanStack Query /
Recharts stack - all pre-approved in the spec's own STACK section, so no fresh dependency
approval was needed for the frontend framework itself. Three real gaps needed resolving before
any of the three surfaces could be built for real, all raised with the user directly first.

## Decision 1 — live traffic: an in-memory ring buffer + SSE, not Postgres

Nothing built in M0-M9 ever wired up the Postgres service `ops/compose.yaml` scaffolded back at
Milestone 0 for "decision log, embedding store" - every milestone's gateway work returned a
decision per request and moved on, never persisting it anywhere. For "live traffic - streaming
decision feed" to be real rather than simulated, something has to hold recent decisions and push
them to the dashboard.

Raised directly: build real Postgres persistence now (schema, migrations, a write path into the
gateway - real, durable, but real added scope), or an in-memory ring buffer with Server-Sent
Events (no new infra dependency, matches L4's own "ephemeral is fine, TTL-friendly" precedent,
ADR-0008)? **Decision: in-memory + SSE.** `packages/gateway/src/portcullis/gateway/decision_log.py`
holds the last 500 decisions in a bounded deque and fans them out to connected SSE clients
(`GET /v1/decisions/stream`); `GET /v1/decisions/recent` serves the current buffer for a page
that just loaded. This is real - the feed shows genuine decisions from genuine requests, the same
honesty standard every other milestone's demo held to - just not durable across a gateway
restart, and not queryable historically. A real decision log (Postgres, as the stack always
named) is real future work, not simulated here.

## Decision 2 — the threshold slider: precomputed scores, client-side recomputation

"A threshold slider that recomputes the confusion matrix against the eval set in real time" needs
per-row `(label, score)` pairs, not the aggregate metrics `eval-results.json` (M9) already
carries. `packages/eval/src/portcullis/eval/harness.py` was extended to also dump these ~20k
pairs to `docs/benchmarks/eval-scores.json`; a new gateway endpoint (`GET /v1/eval/scores`) loads
that file once at startup and serves it from memory. Sliding the threshold itself happens
entirely client-side (a fixed array of scores, one pass to recompute TP/FP/TN/FN counts) - no
backend round trip per drag, which is what "real time" actually requires; the backend's job is
handing over the real numbers once, not re-scoring on every pixel of slider movement.

## Decision 3 — "diff two model versions": M9's ablation, not a second trained model

Only one trained L2 checkpoint exists (Milestone 4). Rather than train a second model purely to
have something to diff - a real, non-trivial commitment of GPU-less CPU training time for a
feature whose point is the *diff UI*, not a second model's quality - the red-team console diffs
the two fusion variants Milestone 9 already fit and scored for real: full fusion (with kNN) and
the `remove_knn` ablation. `GET /v1/eval/ablations` serves both variants' scores and aggregate
metrics, labelled honestly as an ablation comparison ("fusion with kNN" vs. "fusion without kNN"),
not presented as two different model generations. This reuses real M9 data end to end rather than
fabricating a second model version to make the UI feature look more impressive than what actually
exists.

## Decision 4 — CORS, added without a fresh ask

The dashboard (its own Next.js dev server, a different origin) needs to call the gateway's API
directly. `fastapi.middleware.cors.CORSMiddleware` ships with FastAPI/Starlette already -
enabling it is a few lines of configuration, not a new dependency, so this was added without a
separate dependency question the way `httpx`/`uvicorn`/`redis` were at M6-M7.

## Decision 5 — the frontend's own quality bar

TypeScript strict (`tsc --noEmit` clean, matching `mypy --strict`'s role on the Python side),
ESLint clean, and unit tests (Vitest) for the pure logic that has no business being untested (the
confusion-matrix-from-threshold computation, the decision-taxonomy aggregation) - the same "test
detection-adjacent logic" instinct the working agreement asks for on the Python side, applied
here to the TypeScript that computes real numbers on screen. UI correctness itself is verified the
way every prior milestone's demo was verified: driving the real running app in a real browser
against the real running gateway, not a mocked API layer - screenshots and the milestone's own
named checkpoint (a demo GIF) come from that real session, not a storyboard.

## What Milestone 10 does not include

- Durable, historical decision logging (Postgres) - Decision 1's real, disclosed gap.
- A genuinely second trained model version to diff - Decision 3's real, disclosed gap.
- Authentication/multi-tenant access control on the dashboard or the new gateway endpoints - a
  single-operator demo surface, matching every other milestone's checkpoint scope.
- Any change to the detection cascade itself - this milestone makes existing results legible, it
  does not add or change a layer.
