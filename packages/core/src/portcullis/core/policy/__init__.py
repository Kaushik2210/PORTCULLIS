"""The policy engine: ALLOW | FLAG | SANITISE | CHALLENGE | BLOCK over a
calibrated fusion score, with shadow mode as a first-class option.
"""

from .policy import PolicyConfig, PolicyResult, Verdict, decide

__all__ = ["PolicyConfig", "PolicyResult", "Verdict", "decide"]
