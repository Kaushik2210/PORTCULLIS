"""Pydantic v2 request/response models for both API surfaces (spec's API
CONTRACT section): the standalone detection API and the OpenAI-compatible
chat-completions proxy.

The chat-completion models are a deliberate subset of OpenAI's real schema -
enough fields to proxy real client traffic (model, messages, stream), not a
full reimplementation of every sampling parameter. Extra fields a client
sends that aren't modelled here are accepted and passed through raw by the
proxy layer rather than validated, so a real OpenAI client's full parameter
set doesn't get rejected at the gateway's door.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from portcullis.core.l4 import ConversationState
from portcullis.core.policy import Verdict

VerdictStr = Literal["allow", "flag", "sanitise", "challenge", "block"]
ConversationStateStr = Literal["normal", "probing", "establishing", "exploiting"]


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    scope: Literal["user", "system", "retrieved", "tool_result", "file"] = "user"
    shadow: bool = False
    conversation_id: str | None = None
    """Opt-in multi-turn tracking (L4, ADR-0008). Omitted (the default)
    reproduces exactly Milestone 6's stateless behaviour - this field did
    not exist then, and its absence now means nothing changes for an
    existing caller."""


class MatchedRule(BaseModel):
    rule_id: str
    name: str
    severity: str
    labels: tuple[str, ...]
    weight: float
    span: tuple[int, int]
    matched_text: str
    rationale: str


class LatencyBreakdownModel(BaseModel):
    l0_ms: float
    l1_ms: float
    l2_ms: float
    knn_ms: float
    fusion_ms: float
    total_ms: float


class DetectResponse(BaseModel):
    verdict: VerdictStr
    enforced: bool
    score: float
    rationale: str
    contributions: dict[str, float]
    matched_rules: tuple[MatchedRule, ...]
    taxonomy_labels: tuple[str, ...]
    obfuscation_score: float
    max_decode_depth: int
    nearest_known_attack: str | None
    nearest_known_attack_family: str | None
    latency_ms: LatencyBreakdownModel
    conversation_state: ConversationStateStr | None = None


class BatchDetectRequest(BaseModel):
    items: tuple[DetectRequest, ...] = Field(min_length=1, max_length=256)


class BatchDetectResponse(BaseModel):
    results: tuple[DetectResponse, ...]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    conversation_id: str | None = None
    """Same opt-in L4 tracking as `DetectRequest.conversation_id` - not a
    real OpenAI API field, but additive: a real OpenAI client that never
    sets it is unaffected, and this field is stripped before the request is
    forwarded upstream (proxy.py never sees it)."""


class BlockedErrorDetail(BaseModel):
    message: str
    type: Literal["portcullis_block"] = "portcullis_block"
    code: Literal["blocked"] = "blocked"
    portcullis: DetectResponse


class BlockedErrorResponse(BaseModel):
    error: BlockedErrorDetail


class EgressBlockedErrorDetail(BaseModel):
    """A confirmed canary leak (ADR-0009) - a different event from
    `BlockedErrorDetail`, which is an input-side policy verdict. This one
    carries no `DetectResponse`: a canary match is certainty, not a
    calibrated probability, so there's no fusion score or policy rationale
    to attach - only the fact of the leak and why it was recognised."""

    message: str
    type: Literal["portcullis_egress_block"] = "portcullis_egress_block"
    code: Literal["canary_leak"] = "canary_leak"
    rationale: str


class EgressBlockedErrorResponse(BaseModel):
    error: EgressBlockedErrorDetail


_VERDICT_TO_STR: dict[Verdict, VerdictStr] = {
    Verdict.ALLOW: "allow",
    Verdict.FLAG: "flag",
    Verdict.SANITISE: "sanitise",
    Verdict.CHALLENGE: "challenge",
    Verdict.BLOCK: "block",
}


def verdict_to_str(verdict: Verdict) -> VerdictStr:
    return _VERDICT_TO_STR[verdict]


_CONVERSATION_STATE_TO_STR: dict[ConversationState, ConversationStateStr] = {
    ConversationState.NORMAL: "normal",
    ConversationState.PROBING: "probing",
    ConversationState.ESTABLISHING: "establishing",
    ConversationState.EXPLOITING: "exploiting",
}


def conversation_state_to_str(state: ConversationState | None) -> ConversationStateStr | None:
    return None if state is None else _CONVERSATION_STATE_TO_STR[state]
