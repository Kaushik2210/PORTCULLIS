"""Adaptive-attacker evaluation (ADR-0010, Decision 5): two dependency-free
techniques, both assuming grey/black-box query access to the deployed
system (ADR-0001's adversary (b)/(c)) - paraphrase attacks (cheap mechanical
rephrasing) and iterative black-box query attacks (greedy hill-climbing
against the fused score). Both report attack success rate honestly,
including where it's bad.
"""

from .blackbox_query import BlackBoxAttackResult, run_blackbox_attack
from .paraphrase import ParaphraseAttempt, generate_paraphrases

__all__ = [
    "BlackBoxAttackResult",
    "ParaphraseAttempt",
    "generate_paraphrases",
    "run_blackbox_attack",
]
