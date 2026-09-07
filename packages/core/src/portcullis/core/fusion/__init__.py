"""L5-independent fusion: combine per-layer scores into one calibrated
probability, and hold that combination to the honesty check ADR-0001 sets
for it - compare against a hand-tuned max() and say if fusion doesn't win.

Dependency-light by design: this package applies fitted weights, it does not
fit them. Fitting (packages/training/.../fusion/fit.py) needs the real
corpus and scikit-learn; applying needs neither.
"""

from .fuse import fuse, max_baseline
from .types import FusionResult, FusionWeights, LayerScores, MaxBaseline

__all__ = [
    "FusionResult",
    "FusionWeights",
    "LayerScores",
    "MaxBaseline",
    "fuse",
    "max_baseline",
]
