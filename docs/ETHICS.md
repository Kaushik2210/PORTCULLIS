# Ethics and responsible disclosure

## What this repository is

PORTCULLIS trains and evaluates a **defence**. To do that it necessarily contains a corpus of
prompt-injection and jailbreak attempts, because a detector cannot be built or honestly measured
without adversarial examples.

The corpus aggregates **already-published research artifacts** — public datasets released by
academic groups and security researchers, under their stated licences, with provenance recorded
per row group in the dataset manifest. It contributes no novel attack techniques.

## What this repository is not

**Not a jailbreak how-to.** The corpus is organised for classifier training and family-disjoint
evaluation, not for use. It is not curated, ranked, or annotated by effectiveness against any
production system, and this repository publishes no guidance on applying it to one.

**Not a content-moderation system.** PORTCULLIS detects instruction subversion, not harmful
content. See the threat model, section 5.

## Responsible disclosure

Novel working bypasses discovered while developing or red-teaming this project go through
**responsible disclosure to the affected vendor**, not into the README, the commit log, or a
conference talk.

This applies specifically to:

- Bypasses of a third-party model's safety training discovered while assembling the corpus.
- Bypasses of a commercial detection product found while running it as an evaluation baseline.
- Injection techniques that prove effective against a deployed system the author does not own.

Where a technique must be described to justify a detection rule, the repository describes the
**detected pattern** and its rationale, not a working payload tuned against a named target.
Rule test fixtures are minimal reproductions sufficient to test the rule.

Bypasses of **PORTCULLIS itself** are in scope for public discussion and are actively sought.
The adaptive-attacker evaluation exists to find them, and its results are published including
the failures. A defensive project that reports only its wins is not reporting.

## Corpus handling

- **Manifests and hashes only.** Raw corpora are never committed. `data/` carries manifests with
  content hashes; `just eval` reconstructs the corpus from its sources.
- **Licence tags per source.** Every row group records its source and licence. Sources whose
  licence does not permit redistribution are referenced, never vendored.
- **Gated sources are opt-in.** Datasets behind a click-through agreement are excluded from the
  default reproducible path and gated behind an explicit flag. Published numbers record which
  data tier produced them. See [ADR-0004](adr/0004-corpus-and-gated-data.md).
- **Deletion requests are honoured.** Where a source licence reserves the right to require
  deletion, that source is Tier 2 and can be dropped without invalidating the default pipeline.

## Evaluation baselines

Commercial detection APIs are evaluated as baselines under their published terms of service and
free tiers. Results are reported factually, including cases where a baseline outperforms
PORTCULLIS. Baselines are not stress-tested for vulnerabilities, and no attempt is made to
characterise a commercial product's internals beyond its documented behaviour.

## Intended use

This project is intended for defenders: teams operating LLM applications who need to detect
instruction subversion in their own traffic, and researchers evaluating detection approaches.
Deploying it against traffic you are not authorised to inspect is out of scope and not
supported.
