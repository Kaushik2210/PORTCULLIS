# PORTCULLIS dashboard

Next.js 15 App Router / TypeScript strict / Tailwind v4 / shadcn/ui / TanStack Query / Recharts.
Three surfaces per the project spec — live traffic, decision inspector, red-team console. Design
rationale (the in-memory decision feed, the threshold slider's client-side computation, the
ablation-as-model-diff decision) is in
[`docs/adr/0011-dashboard-scope-and-data-sources.md`](../../docs/adr/0011-dashboard-scope-and-data-sources.md);
what actually got verified working is in
[`docs/benchmarks/dashboard-report.md`](../../docs/benchmarks/dashboard-report.md).

## Running it

Needs the gateway running separately (`just gateway-serve` + `just gateway-mock-upstream`, or any
of the `just demo-m*` recipes) - `NEXT_PUBLIC_GATEWAY_URL` overrides the default
`http://127.0.0.1:8001` if the gateway is running somewhere else.

```bash
just dashboard        # from the repo root - starts the Next.js dev server
```

or, from this directory: `npm run dev`.

## Checks

```bash
just dashboard-check  # from the repo root - tsc --noEmit, eslint, vitest
```

This is the frontend's equivalent of the Python side's `just check` (ruff, mypy --strict,
pytest). `src/lib/confusion-matrix.ts` is the one piece of pure client-side logic with its own
unit tests (`src/lib/__tests__/`) - the same "test detection-adjacent logic" instinct the Python
side applies, here to the code that computes real confusion-matrix numbers on screen.

## Layout

```
src/app/live/         Live traffic: decision feed (SSE), taxonomy breakdown, threshold slider
src/app/inspector/    Decision inspector: paste a prompt, watch it descend the cascade
src/app/red-team/     Red-team console: M9 replay results, ablation diff, report export
src/lib/api.ts         Typed fetch wrappers for the gateway - mirrors schemas.py exactly
src/lib/confusion-matrix.ts   Pure confusion-matrix computation, unit tested
src/components/ui/     shadcn/ui primitives
```
