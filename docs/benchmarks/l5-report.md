# L5 egress inspection — Milestone 8

What got built, what the live demo actually caught, and a real bug the gateway test suite found
before a person did. Design rationale (canary statelessness, the redact-vs-block split, the
sliding-window trade-off, the mock upstream's deliberate vulnerability) is in
[ADR-0009](../adr/0009-l5-egress-inspection.md); this report is the measurement side of it.

## What was built

- `packages/core/src/portcullis/core/l5/` - pure scanning logic, no ML dependency:
  - `canary.py` - generate a high-entropy per-request canary, inject it into the outbound system
    prompt, scan a response for its exact (never fuzzy) reappearance.
  - `secrets.py` - named-pattern rules (AWS keys, generic API-key prefixes, PEM private-key
    headers, JWTs, email, phone) plus a Shannon-entropy fallback for shapes no named rule covers.
  - `exfil.py` - markdown-image URLs carrying a long, high-entropy query value; tool-call-shaped
    syntax appearing in free text.
  - `stream_scanner.py` - `SlidingWindowScanner`, a sliding-window scan over incrementally
    arriving text so a match split across SSE chunks is still caught.
  - `scanner.py` - combines all three into one `scan()`/`redact()` pass over a complete response.
- `packages/gateway/src/portcullis/gateway/egress.py` - wires L5 into the proxy: injects the
  canary into the outbound payload, scans the buffered or streamed response, applies ADR-0009's
  differentiated action (canary -> halt; secret/exfil -> redact and continue).
- `mock_upstream.py` gained one deliberate vulnerability (ADR-0009, Decision 5): a recognisable
  extraction phrase makes it echo its system message back, canary included - simulating the real
  failure class canary tokens exist to catch, disclosed as a simulation rather than hidden.

## Testing

37 pure unit tests for the scanning logic itself (`test_l5_canary.py`, `test_l5_secrets.py`,
`test_l5_exfil.py`, `test_l5_scanner.py`, `test_l5_stream_scanner.py`), plus 6 gateway-level
tests (`test_egress.py`) against the real HTTP round trip through the mock upstream via
`ASGITransport` - no mocking of the scan logic itself, so a canary leak in those tests is a
genuine leak the mock upstream produced, caught by the same code a real provider's response would
go through.

**A real bug the sliding-window tests caught in themselves, before it shipped:** the first
version of `SlidingWindowScanner` released text once it aged past the window boundary, but the
window (`window_chars`) was smaller than the pattern being protected in two of the tests written
against it - a 51-character canary against a 10-character window, a 17-character email against
the same. Tracing through the release logic showed this isn't a test-only problem: **a match
longer than the window gets its own leading characters released as plain text before the pattern
completes and is recognised**, silently defeating the entire point of Decision 4. Fixed two ways:
the tests were wrong to use a window smaller than what they were testing, so they were corrected
to use a properly-sized window; and, because the canary's length is fully known (this code
generates it), `SlidingWindowScanner.__init__` now refuses construction outright if
`window_chars` is smaller than the canary, converting a silent leak into a loud
misconfiguration error. Secret/exfil patterns have no such length guarantee (a JWT or a
multi-line private key block can be arbitrarily long) - that limitation is real, disclosed in the
module's own docstring, and not guarded against, since there is no fixed bound to guard against.

**A second real bug, found by the gateway test suite (not a unit test) on the very first
end-to-end run:** `inject_canary()` checked `isinstance(raw_messages, list)` before rebuilding
the outbound messages, but `ChatCompletionRequest.model_dump()` preserves the field's declared
container type - `messages: tuple[ChatMessage, ...]` dumps as a Python `tuple`, not a `list`. The
check silently matched `False`, the `else` branch ran, and the entire message history - including
the user's actual message - was replaced with just the injected system message. Every
`/v1/chat/completions` test that checked the echoed reply's content failed immediately
(`'hello there' in 'Mock response to: '` - the reply was empty). Fixed by checking
`isinstance(raw_messages, list | tuple)`. Caught before any manual testing, by the existing M6/M7
test suite simply continuing to run against the new code - exactly what that suite is for.

## What the live demo actually caught

Run for real (`just demo-m8`) against the trained M4/M5 checkpoint and the (deliberately
vulnerable) mock upstream:

```
--- benign request (canary injected, invisible) ---
status: 200
reply:  Mock response to: Can you help me reset my password?

--- extraction attempt (mock genuinely leaks its system prompt) ---
status: 400
blocked: Response blocked by PORTCULLIS: confirmed system-prompt extraction detected in the
model's reply.
rationale: The internal tracking canary injected into this request's system prompt appeared
verbatim in the model's response - a confirmed system-prompt extraction.

--- message containing a fake AWS key (echoed back by the mock) ---
status: 200
reply:  Mock response to: My AWS key is [REDACTED:aws-access-key], can you note that down?
```

The middle case is the milestone's own named checkpoint: a real leak (the mock upstream, made
vulnerable on purpose per ADR-0009 Decision 5, genuinely echoed its system prompt including the
canary) caught by the same scanning code a real provider's identical failure would go through.
The third case shows the *other* half of ADR-0009's design: a probable-but-uncertain finding
(an AWS-key-shaped string) does not take down the whole response - it gets redacted and the
reply still reaches the client.

## What Milestone 8 does not establish

- Any benchmarked false-positive rate for the secret/PII/exfiltration rules - each rule has its
  own true/false-positive test cases (the same discipline L1's rules follow), not a measurement
  against a labelled corpus. No such corpus exists for this layer any more than one exists for L4
  (ADR-0008).
- A measured time-to-first-token cost for the sliding window - `DEFAULT_WINDOW_CHARS = 200` is a
  reasoned default, disclosed as unmeasured in ADR-0009 Decision 4. Milestone 9's harness is
  where a real number would come from.
- Any guarantee against secrets/patterns longer than the configured window - a real, disclosed
  limitation (this report's testing section, and `stream_scanner.py`'s own docstring), not solved
  here.
- Wiring L5 into `/v1/detect` - by design; that endpoint scores a single piece of text, not a
  proxied response. See ADR-0009's "what this milestone does not include" section.
