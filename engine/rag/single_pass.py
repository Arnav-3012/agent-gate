"""Single-pass RAG: one retrieval, one LLM call. The cheap arm of the router."""
import time

from engine.llm.client import LLMClient, Message
from engine.rag.retriever import Hit, Retriever
from engine.tracing import schema as ev
from engine.tracing.emitter import Emitter

SYSTEM = (
    "Answer the question using only the context. Reply with the shortest possible answer "
    "(a name, date, number, or yes/no), no full sentence. If the context lacks the answer, "
    "give your best guess."
)


FINANCE_SYSTEM = (
    "Answer the question using only the context from financial filings. Start with the final "
    "answer (include units and the fiscal period, e.g. '$1,577 million in FY2018'), then add at "
    "most two short sentences of justification. If the context lacks the answer, give your best "
    "estimate and say it is an estimate."
)


def build_messages(question: str, hits: list[Hit], system: str = SYSTEM) -> list[Message]:
    ctx = "\n\n".join(f"[{h.chunk.title}] {h.chunk.text}" for h in hits)
    return [
        Message(role="system", content=system),
        Message(role="user", content=f"Context:\n{ctx}\n\nQuestion: {question}"),
    ]


def run_rag(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    emitter: Emitter,
    max_answer_tokens: int = 64,
    system: str = SYSTEM,
) -> str:
    """Emits retrieve(step 0) -> answer -> stop. Failures end in stop(reason="error"); returns "" then."""
    t0 = time.perf_counter()
    try:
        hits = retriever.search(question)
        emitter.emit(
            ev.Retrieve,
            step=0,
            query=question,
            hits=[ev.Hit(chunk_id=h.chunk.chunk_id, title=h.chunk.title, score=h.score) for h in hits],
            latency_s=time.perf_counter() - t0,
        )
        call = llm.chat(build_messages(question, hits, system), max_tokens=max_answer_tokens)
    except Exception as e:  # noqa: BLE001 - any backend failure must still terminate the trace
        emitter.emit(
            ev.Stop, reason="error", steps=1, total_tokens=llm.meter.total_tokens,
            total_latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}",
        )
        return ""
    text = call.text.strip()
    emitter.emit(
        ev.Answer,
        text=text,
        llm=ev.LLMStats(
            prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens,
            cached_tokens=call.cached_tokens, latency_s=call.latency_s,
        ),
    )
    emitter.emit(
        ev.Stop, reason="rag_done", steps=1, total_tokens=call.total_tokens,
        total_latency_s=time.perf_counter() - t0,
    )
    return text
