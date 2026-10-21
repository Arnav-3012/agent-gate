"""Capped agentic RAG: plan once, then retrieve -> reflect per step, then one answer call."""
import re
import time
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from engine.llm.client import LLMClient, Message
from engine.rag.retriever import Hit, Retriever
from engine.rag.single_pass import build_messages
from engine.tracing import schema as ev
from engine.tracing.emitter import Emitter

M = TypeVar("M", bound=BaseModel)

PLAN_SYSTEM = (
    "Break the question into at most 3 short search queries, in the order they should be "
    'searched (later ones may depend on earlier answers). Reply with JSON only: '
    '{"subqueries": ["..."]}'
)
REFLECT_SYSTEM = (
    "Decide whether the evidence is enough to answer the question. Reply with JSON only: "
    '{"confidence": <0..1>, "next_query": "<what to search next, or empty if enough>"}'
)


class AgentConfig(BaseModel):
    max_steps: int = 4
    max_tokens: int = 6000
    confidence_threshold: float = 0.8
    min_new_evidence: int = 1
    top_k: int = 5
    max_answer_tokens: int = 64

    @classmethod
    def from_yaml(cls, path: Path = Path("config/engine.yaml")) -> "AgentConfig":
        raw = yaml.safe_load(path.read_text())
        stop = raw.get("stop", {})
        return cls(
            max_steps=raw["max_steps"], max_tokens=raw["max_tokens_per_query"], top_k=raw["top_k"],
            confidence_threshold=stop["confidence_threshold"],
            min_new_evidence=stop["min_new_evidence"],
        )


class _Plan(BaseModel):
    subqueries: list[str] = Field(min_length=1, max_length=3)


class _Reflection(BaseModel):
    confidence: float = Field(ge=0, le=1)
    next_query: str = ""


def _ask_json(llm: LLMClient, messages: list[Message], model: type[M]) -> tuple[M, ev.LLMStats]:
    """One structured LLM call, retried once on bad JSON; stats cover all attempts."""
    stats = ev.LLMStats(prompt_tokens=0, completion_tokens=0, latency_s=0.0)
    for attempt in range(2):
        call = llm.chat(messages, max_tokens=128)
        stats.prompt_tokens += call.prompt_tokens
        stats.completion_tokens += call.completion_tokens
        stats.cached_tokens += call.cached_tokens
        stats.latency_s += call.latency_s
        blob = re.search(r"\{.*\}", call.text, re.DOTALL)  # tolerate fences / chatter
        try:
            return model.model_validate_json(blob.group(0) if blob else ""), stats
        except ValidationError:
            if attempt == 1:
                raise ValueError(f"unparseable {model.__name__} reply: {call.text[:80]!r}") from None
    raise AssertionError("unreachable")


def _evidence_text(evidence: dict[str, Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.title}] {h.chunk.text}" for h in evidence.values())


def run_agent(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    emitter: Emitter,
    cfg: AgentConfig | None = None,
) -> str:
    """Emits plan -> (retrieve -> reflect)* -> answer -> stop. Returns "" on error (stop=error)."""
    cfg = cfg or AgentConfig()
    t0 = time.perf_counter()
    tokens0 = llm.meter.total_tokens

    def used() -> int:
        return llm.meter.total_tokens - tokens0

    evidence: dict[str, Hit] = {}  # scratch memory: every chunk seen, first hit wins
    steps, reason = 0, "max_steps"
    step_cost = answer_reserve = 0  # last step's tokens; ~size of the answer prompt + decode
    try:
        plan, stats = _ask_json(llm, [
            Message(role="system", content=PLAN_SYSTEM), Message(role="user", content=question),
        ], _Plan)
        emitter.emit(ev.Plan, step=0, subqueries=plan.subqueries, llm=stats)
        queue = list(plan.subqueries)
        while steps < cfg.max_steps:
            if used() + step_cost + answer_reserve > cfg.max_tokens:  # projected, not just spent
                reason = "max_tokens"
                break
            steps += 1
            before = used()
            query = queue.pop(0)
            t_ret = time.perf_counter()
            hits = retriever.search(query, top_k=cfg.top_k)
            emitter.emit(
                ev.Retrieve, step=steps, query=query,
                hits=[ev.Hit(chunk_id=h.chunk.chunk_id, title=h.chunk.title, score=h.score)
                      for h in hits],
                latency_s=time.perf_counter() - t_ret,
            )
            fresh = [h for h in hits if h.chunk.chunk_id not in evidence]
            evidence.update({h.chunk.chunk_id: h for h in fresh})
            refl, stats = _ask_json(llm, [
                Message(role="system", content=REFLECT_SYSTEM),
                Message(role="user",
                        content=f"Question: {question}\n\nEvidence:\n{_evidence_text(evidence)}"),
            ], _Reflection)
            next_query = refl.next_query.strip()
            if next_query:
                queue.insert(0, next_query)
            step_cost = used() - before
            answer_reserve = stats.prompt_tokens + cfg.max_answer_tokens
            if refl.confidence >= cfg.confidence_threshold:
                reason = "confidence"
            elif len(fresh) < cfg.min_new_evidence or not queue:
                reason = "no_new_evidence"
            emitter.emit(
                ev.Reflect, step=steps, confidence=refl.confidence, new_evidence=len(fresh),
                next_query=next_query, verdict="stop" if reason != "max_steps" else "continue", llm=stats,
            )
            if reason != "max_steps":
                break
        call = llm.chat(build_messages(question, list(evidence.values())),
                        max_tokens=cfg.max_answer_tokens)
    except Exception as e:  # noqa: BLE001 - any backend/parse failure must still terminate the trace
        emitter.emit(
            ev.Stop, reason="error", steps=steps, total_tokens=used(),
            total_latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}",
        )
        return ""
    text = call.text.strip()
    emitter.emit(
        ev.Answer, text=text,
        llm=ev.LLMStats(
            prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens,
            cached_tokens=call.cached_tokens, latency_s=call.latency_s,
        ),
    )
    emitter.emit(
        ev.Stop, reason=reason, steps=steps, total_tokens=used(),
        total_latency_s=time.perf_counter() - t0,
    )
    return text
