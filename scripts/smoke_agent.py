"""Phase 2 agent check: same question through RAG and the agent; compare answer, tokens, latency."""
import json
import sys
import uuid
from pathlib import Path

from engine.agent.loop import AgentConfig, run_agent
from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Retriever
from engine.rag.single_pass import run_rag
from engine.tracing.emitter import Emitter, JsonlSink


def main() -> None:
    rows = [json.loads(l) for l in Path("eval/data/questions.jsonl").read_text().splitlines()]
    bridge = next(r for r in rows if r["type"] == "bridge")
    q = sys.argv[1] if len(sys.argv) > 1 else bridge["question"]
    gold = bridge["answer"] if len(sys.argv) == 1 else "?"
    r = Retriever()
    pricing = Pricing.from_yaml()
    for name, fn in (("rag", run_rag), ("agent", None)):
        em = Emitter(uuid.uuid4().hex, [JsonlSink(Path(f"traces/smoke_{name}.jsonl"))], Ledger(pricing))
        llm = LLMClient()
        ans = run_rag(q, r, llm, em) if fn else run_agent(q, r, llm, em, AgentConfig.from_yaml())
        stop = em.events[-1]
        print(f"\n[{name}] A: {ans!r} (gold {gold!r})\n  events={[e.type for e in em.events]}")
        print(f"  stop={stop.reason} steps={stop.steps} tokens={stop.total_tokens} "
              f"latency={stop.total_latency_s:.1f}s cost=${stop.total_cost_usd:.7f}")
        for e in em.events:
            if e.type in ("plan", "retrieve", "reflect"):
                print("   ", e.type, getattr(e, "subqueries", None) or getattr(e, "query", None)
                      or f"conf={e.confidence} new={e.new_evidence}")


if __name__ == "__main__":
    main()
