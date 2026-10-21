"""Build the finance corpus + questions from FinanceBench (150 open-source Qs over SEC filings).

Corpus = every evidence page, whitespace-collapsed (raw pages are full of "\\n \\n$\\n" noise that
inflates tokens) and split into ~900-char overlapping windows, so a query sends ~1.5k tokens
instead of 4-6k. Each question's distractors are the other questions' pages (same companies/years
included). Gold chunks = the windows overlapping that question's evidence snippet (found verbatim
in its page for all 189 snippets). Outputs eval/data/finance_questions.jsonl and the Qdrant collection "finance".
Question `type` = FinanceBench question_type (metrics-generated | domain-relevant | novel-generated).
"""
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from engine.rag.retriever import Chunk, Embedder, make_chunk, point_id

COLLECTION = "finance"
QUESTIONS = Path("eval/data/finance_questions.jsonl")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def windows(text: str, size: int = 900, overlap: int = 150) -> list[tuple[int, int]]:
    """Character spans of overlapping windows that end at a space (never mid-number)."""
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text) and (cut := text.rfind(" ", start + size // 2, end)) > 0:
            end = cut
        spans.append((start, end))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return spans


def build() -> tuple[list[dict], list[Chunk]]:
    ds = load_dataset("PatronusAI/financebench", split="train")
    chunks: dict[str, Chunk] = {}
    questions = []
    for r in ds:
        gold = []
        for e in r["evidence"]:
            title = f"{r['company']} {r['doc_period']} {r['doc_type']} p.{e['evidence_page_num'] + 1}"
            page, snippet = squash(e["evidence_text_full_page"]), squash(e["evidence_text"])
            at = page.find(snippet) if snippet else -1
            for a, b in windows(page):
                c = make_chunk(title, page[a:b], "financebench")
                chunks[c.chunk_id] = c
                if at < 0 or (a < at + len(snippet) and b > at):  # no span found: whole page is gold
                    gold.append(c.chunk_id)
        questions.append({
            "id": r["financebench_id"], "source": "financebench", "type": r["question_type"],
            "level": r["question_reasoning"] or "n/a", "question": r["question"],
            "answer": r["answer"], "gold_chunk_ids": sorted(set(gold)),
        })
    return questions, sorted(chunks.values(), key=lambda c: c.chunk_id)


def main() -> None:
    questions, chunks = build()
    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS.write_text("".join(json.dumps(q) + "\n" for q in questions))
    emb = Embedder()
    vectors = emb.passages([c.embed_text for c in chunks])
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    client.recreate_collection(
        COLLECTION, vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE)
    )
    client.upsert(COLLECTION, [PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump())
                               for c, v in zip(chunks, vectors)])
    print(f"questions={len(questions)} chunks={len(chunks)} -> {QUESTIONS}, qdrant:{COLLECTION}")


if __name__ == "__main__":
    main()
