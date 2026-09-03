# ADR-0003 — FAISS for the kNN sidecar, not hnswlib

**Status:** Accepted
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
and the dependency is dropped.**

**pgvector.** Already in the stack for the decision log. Rejected for this path: a network round
trip inside the synchronous latency budget, to serve an index that fits comfortably in process
memory.

## Consequences

- The kNN sidecar depends on a binary wheel; CI must verify installation on the target platform,
  not just on Linux.
- `faiss-cpu` has no official type stubs. It will need an entry in the mypy override table, which
  is a small, contained erosion of `--strict`.
- The decision is revisitable and cheap to reverse: the sidecar sits behind an interface, so
  swapping the index implementation touches one adapter.
