# ADR-0007 — Gateway scope: L3 dropped, mock upstream, decision-trace placement

**Status:** Accepted
**Date:** Milestone 6

## Context

Milestone 6 wires L0-L2, the kNN sidecar, fusion and policy (M0-M5) into a live FastAPI gateway:
an OpenAI-compatible transparent proxy plus a standalone detection API. Three scope questions
came up before writing any code, all raised with the user rather than decided silently, per the
working agreement.

## Decision 1 — L3 (LLM adjudication) is dropped, not folded into M6

The architecture section of the original spec describes L3 in detail - an LLM judge that fires
only inside a configurable uncertainty band on the fused score. But the milestone table's 12 rows
(0-11) never gave it one; the table goes straight from M5 (kNN + fusion + policy) to M6 (gateway).
Since the gateway is exactly where L3 would sit - between fusion's score and the policy verdict -
this gap had to be resolved before M6 could be scoped, not discovered partway through.

Raised with the user directly: fold a minimal L3 into M6, give it a dedicated milestone, or cut it
as a documented scope decision. **Decision: cut it.** Reasons that made this the right default to
propose:

- Fusion already emits a calibrated, explainable probability (M5). L3 is meant to *refine*
  uncertain cases, not replace fusion - without it, the uncertainty band simply doesn't get a
  second opinion, which is a capability gap, not a correctness bug.
- An LLM-judge-in-the-loop is a real latency and cost commitment. The project's own NFR-1 budget
  (ADR-0002) is scoped to L0-L2; L3 was always going to need its own SLO discussion, not inherit
  the gateway milestone's.
- L3's prompt is itself an attack surface - the spec calls out that the adjudicator prompt must
  be injection-resistant and ships its own test proving it holds. That is real, separate work
  deserving its own milestone and its own red-team test, not a rider bolted onto the gateway
  milestone to fill a numbering gap.

If a later milestone needs L3, the fused score and policy verdict this milestone produces are
exactly the inputs an uncertainty-band router would need - nothing here forecloses adding it.

## Decision 2 — Mock upstream, not a live provider

`POST /v1/chat/completions` needs something to proxy *to*. Pointing at a real provider (OpenAI,
Anthropic, etc.) means a live API key and real spend on every demo run and every CI invocation of
the gateway's own tests - a cost and credential dependency that should not be decided silently on
the user's behalf.

**Decision:** ship `portcullis.gateway.mock_upstream` - a second, tiny FastAPI app that
implements the same `/v1/chat/completions` shape (streaming and non-streaming) and returns a
deterministic, input-derived reply (`"Mock response to: {last user message}"`) rather than a
canned string, so the demo visibly shows the round trip reflects real input. It is a genuine
second HTTP hop, not a function call standing in for one - the gateway's proxy code calls it over
real HTTP via `httpx`, exercising the same relay code path a live provider would need.

This keeps the M6 checkpoint (`base_url` swap demo) fully reproducible with zero external
dependencies: point a client at the mock upstream directly, see the raw behaviour; point the same
client at the gateway instead, see detection applied on top, with an injection attempt blocked
before it ever reaches the upstream. Swapping in a real provider later is a one-line config change
(`PORTCULLIS_UPSTREAM_BASE_URL`) once the user chooses to authorise that spend - this ADR does not
foreclose it, it just declines to assume it.

## Decision 3 — Where the decision trace lives

The spec says "blocked requests return a structured error with the decision trace attached." For
allowed/flagged/shadow-mode traffic, the response body needs to stay OpenAI-shaped for the
`base_url` swap to actually be transparent - injecting a `portcullis` field into a successful
`ChatCompletionResponse` risks breaking a strictly-typed OpenAI client that doesn't expect it.

**Decision:**
- **Blocked requests** (`enforced=True`, verdict `BLOCK`): the response body *is* a structured
  error (OpenAI's own `{"error": {...}}` shape) with the full decision trace nested under
  `error.portcullis` - exactly what the spec asks for, and there is no successful body to protect
  the shape of.
- **Every other response** (proxied through, streaming or not): the OpenAI-shaped body is left
  untouched; the decision trace rides on `X-Portcullis-Verdict` and `X-Portcullis-Score` response
  headers instead. A caller that wants the full per-layer breakdown for traffic that wasn't
  blocked calls `/v1/detect` directly - that's what the standalone detection API is for.
- **Shadow mode** (`shadow=True`): always proxies through regardless of verdict, per the spec's
  framing of shadow mode as "evaluate and log without enforcing." The would-be verdict still rides
  the response headers, so shadow-mode traffic is fully observable without touching enforcement.

## Two additions to the approved dependency stack

Both raised with the user before use, not added silently:

- **`httpx`** - the spec's stack list names no HTTP client, an omission in the original spec (a
  proxy has to make outbound calls somehow). Already resolved transitively via `datasets`
  (portcullis-training), so declaring it as a direct `portcullis-gateway` and `portcullis-client`
  dependency adds no new download. Chosen over stdlib `urllib`/`http.client` because it is
  async-native and handles SSE streaming relay cleanly; FastAPI/Starlette's own docs recommend it
  as the paired client. Confirmed with the user directly.
- **`uvicorn`** - not named separately from "FastAPI" in the spec's stack list, but there is no
  reasonable alternative: FastAPI is an ASGI application, not a server, and uvicorn is what
  FastAPI's own documentation runs it with. Treated as implied by the already-approved FastAPI
  entry rather than re-asked, since (unlike httpx) there was no real stdlib alternative to weigh -
  disclosed here rather than added quietly.

## What Milestone 6 does not include

- L3, per Decision 1.
- L4 (conversation state, Redis-backed) and L5 (egress inspection) - unbuilt, per the milestone
  table (M7, M8). The gateway's streaming relay has a seam where L5's sliding-window egress scan
  will attach once M8 exists, but nothing scans the stream yet.
- Real-provider proxying - the mock upstream is the M6 default; a live provider is a config change
  the user makes explicitly when they choose to authorise it, per Decision 2.
- Rate limiting and per-tenant/per-route policy config storage (both named in the spec as Redis/
  Postgres concerns) - `PolicyConfig` is instantiated once from environment/config at startup,
  not yet looked up per tenant from a store that doesn't exist yet.
