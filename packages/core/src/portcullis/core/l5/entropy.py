"""Shannon entropy - shared by secrets.py (unknown-shape secret detection)
and exfil.py (high-entropy exfil query values), so exfil.py doesn't need to
depend on secrets.py for one small piece of math both happen to need."""

from __future__ import annotations

import math
from collections import Counter


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    length = len(s)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())
