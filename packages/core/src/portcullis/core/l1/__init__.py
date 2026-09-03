"""L1 - signature and heuristic engine.

Fast, fully explainable, zero ML. Rules live in versioned YAML under rules/ so
they can be reviewed, diffed and shipped without touching Python.

Three properties make this layer worth having despite being the most brittle
detector in the cascade:

* **It says why.** Every match names a rule id, a span in the user's own text,
  and a sentence of rationale. No other layer can do that as cheaply.
* **It is falsifiable.** Every rule ships a true-positive *and* a
  false-positive case, enforced at load time. A pattern nobody can bound is a
  pattern nobody can review.
* **It is scoped.** The same sentence from a user and from a retrieved
  document are different events, and rules say which they mean.
"""

from .dsl import RuleSyntaxError
from .engine import (
    RuleEngine,
    RuleSet,
    load_rules_from_dir,
    load_rules_from_string,
)
from .types import (
    VIEW_CONFIDENCE,
    L1Result,
    Rule,
    RuleMatch,
    RuleTests,
    Scope,
    Severity,
)

__all__ = [
    "VIEW_CONFIDENCE",
    "L1Result",
    "Rule",
    "RuleEngine",
    "RuleMatch",
    "RuleSet",
    "RuleSyntaxError",
    "RuleTests",
    "Scope",
    "Severity",
    "load_rules_from_dir",
    "load_rules_from_string",
]
