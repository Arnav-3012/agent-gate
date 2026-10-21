"""Frozen trace contract. Mirror: dashboard/lib/schema.ts. Change both + SCHEMA_VERSION together."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

SCHEMA_VERSION = 2

Mode = Literal["rag", "agent"]
StopReason = Literal[
    "rag_done", "confidence", "no_new_evidence", "max_steps", "max_tokens", "error"
]


class LLMStats(BaseModel):
    """Metered usage of one LLM call (from engine.llm.client.LLMCall). cost_usd is set by the ledger."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    latency_s: float
    cost_usd: float | None = None


class Hit(BaseModel):
    chunk_id: str
    title: str
    score: float


class _Base(BaseModel):
    trace_id: str
    seq: int  # 0-based, strictly increasing per trace; clients dedupe/poll with it
    ts: float  # unix seconds


class RouteDecided(_Base):
    type: Literal["route_decided"] = "route_decided"
    route: Mode
    p_agent: float | None = None  # classifier probability that agentic is needed
    features: dict[str, float] = {}
    reason: str = ""


class Plan(_Base):
    type: Literal["plan"] = "plan"
    step: int
    subqueries: list[str]
    llm: LLMStats


class Retrieve(_Base):
    type: Literal["retrieve"] = "retrieve"
    step: int  # 0 for single-pass RAG
    query: str
    hits: list[Hit]
    latency_s: float


class Reflect(_Base):
    type: Literal["reflect"] = "reflect"
    step: int
    confidence: float
    new_evidence: int  # chunks not seen in earlier steps
    next_query: str = ""  # what the reflector wants searched next ("" = nothing); v2
    verdict: Literal["continue", "stop"]
    llm: LLMStats


class Escalate(_Base):
    type: Literal["escalate"] = "escalate"
    from_mode: Mode
    to_mode: Mode
    reason: str


class Answer(_Base):
    type: Literal["answer"] = "answer"
    text: str
    llm: LLMStats


class Stop(_Base):
    """Terminal event: a trace is complete once this appears."""

    type: Literal["stop"] = "stop"
    reason: StopReason
    steps: int
    total_tokens: int
    total_latency_s: float
    total_cost_usd: float | None = None
    error: str | None = None


TraceEvent = Annotated[
    RouteDecided | Plan | Retrieve | Reflect | Escalate | Answer | Stop,
    Field(discriminator="type"),
]
event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
