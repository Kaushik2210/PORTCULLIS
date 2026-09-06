"""MinHash / LSH near-duplicate clustering.

Broder (1997), "On the resemblance and containment of documents" - see
NOTICE. This module applies the published technique; nothing here is a novel
algorithm, only the specific shingling and clustering choices for this corpus.

Why near-dup detection matters more than exact-dup detection: two published
benchmarks routinely contain the *same underlying attack, paraphrased*, split
across their own train and test partitions. A row-level or exact-string dedup
misses that entirely. This is the mechanism behind the "disjoint by attack
family" requirement (ADR-0004) as it applies at the row level.
"""

from __future__ import annotations

import re
from typing import Any

from datasketch import MinHash, MinHashLSH

from .schema import Row

# datasketch ships no type stubs (pyproject.toml mypy override), so its
# objects are typed Any in signatures rather than by their real class -
# disallow_any_unimported flags MinHash used as an annotation directly, and
# this is the honest way to say "this is untyped" instead of suppressing it.
MinHashSignature = Any

NUM_PERM = 128
# Word bigrams, threshold 0.5. Chosen from measured Jaccard, not guessed:
# a two-word synonym substitution over a 13-word sentence ("please/kindly",
# "previous/prior") scores 0.6 at k=2, while two genuinely different prompts
# that merely share a common attack preamble ("ignore all previous
# instructions and {tell me a joke / write a poem}") score 0.364 - a real
# margin either side of 0.5, not a coin flip. Trigrams (k=3) collapse that
# margin to 0.467 vs 0.300, both below any threshold that would still catch
# the paraphrase, and 4-grams collapse it further.
SHINGLE_SIZE = 2
SIMILARITY_THRESHOLD = 0.5

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    """Case- and whitespace-insensitive text for shingling.

    Deliberately coarse: dedup is trying to catch "the same attack with a few
    words swapped," not exact byte equality, which the pipeline's manifest
    hashing already handles.
    """
    return _WS.sub(" ", text.lower()).strip()


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = _WORD.findall(_normalize(text))
    if len(words) < k:
        # Too short to shingle meaningfully; fall back to the whole token
        # sequence as one shingle so short rows still get a real signature
        # instead of an empty one that would spuriously "match" every other
        # empty signature.
        return {" ".join(words)} if words else {"\x00__empty__"}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _signature(text: str) -> MinHashSignature:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def _cluster_is_cohesive(
    row_ids: list[str],
    signatures: dict[str, MinHashSignature],
    threshold: float,
    *,
    sample_cap: int = 40,
) -> bool:
    """Verify a provisional cluster is not a chained false merge.

    Union-find over pairwise LSH matches gives single-linkage clustering: if
    A~B and B~C, A and C end up in one cluster even if they share nothing.
    On short, template-heavy prompts this is not a corner case - it happened
    on the real corpus (a "grammar correction tool" template and an
    "are these sentences equivalent" template were merged into one 2478-row
    cluster through intermediate bridge rows, discovered by manual inspection
    during Milestone 3).

    This checks average-linkage instead: for a large cluster, sample pairs
    and require their *mean* estimated Jaccard similarity to still clear the
    threshold. A single-linkage chain typically has plenty of far-apart pairs
    and fails this; a genuine tight cluster (the same template with one
    token substituted) does not. Small clusters are checked exhaustively.
    """
    if len(row_ids) < 2:
        return True

    pairs: list[tuple[str, str]]
    if len(row_ids) <= 12:
        pairs = [
            (row_ids[i], row_ids[j])
            for i in range(len(row_ids))
            for j in range(i + 1, len(row_ids))
        ]
    else:
        # Sample from a fixed seed derived from cluster membership, not a
        # global RNG, so cohesion checking stays deterministic regardless of
        # call order.
        import random

        seed = hash(tuple(sorted(row_ids))) & 0xFFFFFFFF
        rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptography
        pairs = [tuple(rng.sample(row_ids, 2)) for _ in range(sample_cap)]  # type: ignore[misc]

    similarities: list[float] = [signatures[a].jaccard(signatures[b]) for a, b in pairs]
    return (sum(similarities) / len(similarities)) >= threshold


class _UnionFind:
    """Standard union-find with path compression, used to turn LSH's pairwise
    candidate matches into transitive clusters (if a~b and b~c, then a~c)."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def find_clusters(
    rows: list[Row],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
    num_perm: int = NUM_PERM,
) -> dict[str, int]:
    """Return {row.id: cluster_id}. Every row gets exactly one cluster id;
    rows with no near-duplicate get a singleton cluster of their own.

    A provisional cluster that fails the cohesion check (see
    _cluster_is_cohesive) is broken back into singletons rather than kept as
    a merge nobody can justify or repaired by guesswork. Missing a real
    near-duplicate leaves a quality issue - some redundancy in the corpus.
    Wrongly merging distinct rows can delete legitimate data and corrupt the
    leakage audit that other code trusts. The conservative failure mode is
    the only one on offer here.
    """
    if not rows:
        return {}

    signatures = {r.id: _signature(r.text) for r in rows}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for row_id, sig in signatures.items():
        lsh.insert(row_id, sig)

    uf = _UnionFind([r.id for r in rows])
    for row_id, sig in signatures.items():
        for neighbour in lsh.query(sig):
            if neighbour != row_id:
                uf.union(row_id, neighbour)

    provisional: dict[str, list[str]] = {}
    for r in rows:
        provisional.setdefault(uf.find(r.id), []).append(r.id)

    final_groups: list[list[str]] = []
    for members in provisional.values():
        if _cluster_is_cohesive(members, signatures, threshold):
            final_groups.append(members)
        else:
            final_groups.extend([m] for m in members)

    # Deterministic cluster ids: sort groups by their lexicographically
    # smallest member, so numbering does not depend on dict/set iteration
    # order or on how union-find happened to walk the input.
    final_groups.sort(key=min)
    return {row_id: cid for cid, group in enumerate(final_groups) for row_id in group}
