# ADR-0003 — FAISS for the kNN sidecar, not hnswlib

**Status:** Superseded at Milestone 5 — see resolution below. hnswlib remains correctly
rejected; the actual Milestone 5 finding is that *neither* ANN library is needed yet.
**Date:** Milestone 0

## Context

L2 includes an embedding kNN sidecar: sentence-transformer embeddings indexed for nearest-
neighbour lookup against known attack families. It catches novel paraphrases of known families
that the classifier misses, and supplies a "nearest known attack" string that materially improves
explanations.

The original spec named `hnswlib` or FAISS interchangeably. On this project's target platform —
Windows, CPython 3.13 — they are not interchangeable.

## Decision

Use **`faiss-cpu`**.

## Rationale

Verified against PyPI at Milestone 0:

| Package | Latest | Windows / cp313 wheel |
|---|---|---|
| `hnswlib` | 0.8.0 (Dec 2023) | **None — sdist only** |
| `faiss-cpu` | 1.15.0 (Aug 2026) | **Yes** (`faiss_cpu-1.15.0-cp313-cp313-win_amd64.whl`) |

`hnswlib` ships no wheels at all. Installing it on Windows requires a local MSVC C++ toolchain
and compiles a C++ extension at install time. That directly violates the project's
"one-command rebuild on a clean machine" requirement — the failure mode is a compiler error
during `uv sync`, on a contributor's machine, with a message that has nothing to do with this
codebase. Its last release also predates CPython 3.13.

`faiss-cpu` is a maintained wheel on the exact target platform. It is heavier than hnswlib and
brings a larger binary, which is an acceptable cost for an index that is built offline and
memory-mapped at runtime.

Index size is not a differentiator here. The attack-family index is on the order of 10^4–10^5
vectors; both libraries are far below the latency budget at that scale, so the decision is
purely one of packaging reliability.

## Alternatives rejected

**hnswlib.** Rejected on packaging, above. Genuinely faster to build and lighter at small index
sizes; if this project ever ships Linux-only containers as the sole supported path, it is worth
revisiting.

**Exact brute-force search (numpy).** At 10^4–10^5 vectors with ~384-dim embeddings, a full scan
is a single matrix multiply and would be simpler than either library. Worth benchmarking before
assuming an ANN index is needed at all — an honest possibility that this dependency is
unnecessary. Deferred to Milestone 5, where the sidecar is actually built and the comparison
can be measured rather than argued. **If brute force meets the budget, this ADR is superseded
and the dependency is dropped.** — **it did. See Resolution below.**

**pgvector.** Already in the stack for the decision log. Rejected for this path: a network round
trip inside the synchronous latency budget, to serve an index that fits comfortably in process
memory.

## Resolution at Milestone 5

`faiss-cpu` was never installed. The sidecar's real corpus — 6,831 attack rows from the train
partition, 384-dim MiniLM-L6 embeddings (the same checkpoint as L2, ADR-0005 — no second
embedding model) — was benchmarked directly on this machine:

| Method | Latency |
|---|---:|
| Brute-force, single query (numpy `@`, `argpartition` top-k) | 0.69ms |
| Brute-force, batched (200 queries at once) | 0.26ms/query |

Both are two to three orders of magnitude inside any latency tier this project defines (ADR-0002:
Tier B < 25ms, Tier C < 150ms — and this is one component of one layer, not the whole cascade
budget). `faiss-cpu`'s own per-query cost at this corpus size would not be measurably different
— the win FAISS offers is sub-linear scaling as a corpus grows past what fits comfortably in a
brute-force scan, and 6,831 vectors is nowhere near that regime.

**Decision, superseding the one above:** the kNN sidecar (`packages/training/.../knn/index.py`)
is a plain numpy `BruteForceIndex`. No FAISS, no hnswlib, no binary wheel dependency at all.
This ADR's original packaging argument (hnswlib has no Windows/cp313 wheel) stays correct and is
kept above for the record, but it turned out not to matter: the honest question — "is an ANN
index needed at all?" — was the one worth asking, and measurement said no.

**Reopening condition.** If the attack-corpus index grows by one to two orders of magnitude
(hundreds of thousands of vectors) such that a fresh benchmark shows brute-force no longer fits
comfortably inside its layer's latency budget, FAISS is the fallback already evaluated here —
this ADR's original rationale for FAISS over hnswlib remains valid for that scenario.

## Consequences

- No binary-wheel dependency for the kNN sidecar. CI does not need to verify a compiled
  extension's installation on the target platform for this component.
- The decision is cheap to reverse if the corpus grows: `BruteForceIndex` and a hypothetical
  `FaissIndex` can share the same query interface, so swapping the backend touches one adapter,
  not every caller.
- `sentence-transformers` (already in the approved ML stack) is the one new runtime dependency
  this milestone actually added, for embeddings — not an ANN library.
