"""Exit check for Phase 1: question -> top-k chunks -> local LLM answer, plus recall@k on the set."""
import json
import sys
import uuid
from pathlib import Path

from engine.llm.client import LLMClient
from engine.rag.retriever import Retriever
from engine.rag.single_pass import run_rag
from engine.tracing.emitter import Emitter, JsonlSink


def recall_at_k(r: Retriever, rows: list[dict], k: int) -> dict[str, float]:
    """Fraction of gold chunks found in top-k, averaged per source/type."""
    acc: dict[str, list[float]] = {}
    for q in rows:
        got = {h.chunk.chunk_id for h in r.search(q["question"], top_k=k)}
        gold = set(q["gold_chunk_ids"])
        acc.setdefault(q["type"], []).append(len(gold & got) / len(gold))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def main() -> None:
    rows = [json.loads(l) for l in Path("eval/data/questions.jsonl").read_text().splitlines()]
    r = Retriever()
    q = sys.argv[1] if len(sys.argv) > 1 else rows[0]["question"]
    hits = r.search(q)
    for h in hits:
        print(f"{h.score:.3f}  [{h.chunk.title}] {h.chunk.text[:80]}")
    em = Emitter(uuid.uuid4().hex, [JsonlSink(Path("traces/smoke_rag.jsonl"))])
    ans = run_rag(q, r, LLMClient(), em)
    print(f"\nQ: {q}\nA: {ans}\n{[e.type for e in em.events]}  stop={em.events[-1].model_dump()}")
    print("recall@5 by type:", {t: round(v, 2) for t, v in recall_at_k(r, rows, 5).items()})


if __name__ == "__main__":
    main()
