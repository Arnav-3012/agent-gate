"""Build the corpus + eval question set and (re)load Qdrant. Deterministic given --seed.

HotpotQA distractor: each question ships 10 paragraphs (2 gold + 8 distractors), all go in the
corpus. SQuAD: gold paragraph plus random other paragraphs as distractors.
Outputs eval/data/questions.jsonl (gold chunk ids per question) and the Qdrant collection.
"""
import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from engine.rag.retriever import COLLECTION, Chunk, Embedder, make_chunk, point_id

QUESTIONS = Path("eval/data/questions.jsonl")


def hotpot(n_bridge: int, n_comp: int, rng: random.Random) -> tuple[list[dict], dict[str, Chunk]]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    by_type: dict[str, list[int]] = {"bridge": [], "comparison": []}
    for i, t in enumerate(ds["type"]):
        by_type[t].append(i)
    picks = rng.sample(by_type["bridge"], n_bridge) + rng.sample(by_type["comparison"], n_comp)
    chunks: dict[str, Chunk] = {}
    questions = []
    for i in picks:
        r = ds[i]
        by_title = {}
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            c = make_chunk(title, "".join(sents).strip(), "hotpotqa")
            chunks[c.chunk_id] = c
            by_title[title] = c.chunk_id
        gold = sorted({by_title[t] for t in r["supporting_facts"]["title"] if t in by_title})
        questions.append(
            {"id": f"hp-{r['id']}", "source": "hotpotqa", "type": r["type"], "level": r["level"],
             "question": r["question"], "answer": r["answer"], "gold_chunk_ids": gold}
        )
    return questions, chunks


def squad(n_q: int, n_distract: int, rng: random.Random) -> tuple[list[dict], dict[str, Chunk]]:
    ds = load_dataset("rajpurkar/squad", split="validation")
    paragraphs = {c.chunk_id: c for c in (make_chunk(t, x, "squad") for t, x in
                                          dict.fromkeys(zip(ds["title"], ds["context"])))}
    by_ctx = {(c.title, c.text): cid for cid, c in paragraphs.items()}
    picks = rng.sample(range(len(ds)), n_q)
    questions, chunks = [], {}
    for i in picks:
        r = ds[i]
        cid = by_ctx[(r["title"], r["context"])]
        chunks[cid] = paragraphs[cid]
        questions.append(
            {"id": f"sq-{r['id']}", "source": "squad", "type": "single", "level": "easy",
             "question": r["question"], "answer": r["answers"]["text"][0],
             "gold_chunk_ids": [cid]}
        )
    for cid in rng.sample(sorted(paragraphs), n_distract):
        chunks.setdefault(cid, paragraphs[cid])
    return questions, chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bridge", type=int, default=200)
    ap.add_argument("--comparison", type=int, default=100)
    ap.add_argument("--squad", type=int, default=100)
    ap.add_argument("--squad-distractors", type=int, default=1000)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    q_h, c_h = hotpot(a.bridge, a.comparison, rng)
    q_s, c_s = squad(a.squad, a.squad_distractors, rng)
    questions = q_h + q_s
    chunks = sorted({**c_h, **c_s}.values(), key=lambda c: c.chunk_id)

    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS.write_text("".join(json.dumps(q) + "\n" for q in questions))

    emb = Embedder()
    vectors = emb.passages([c.embed_text for c in chunks])
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    client.recreate_collection(
        COLLECTION, vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE)
    )
    points = [PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump())
              for c, v in zip(chunks, vectors)]
    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points[i : i + 256])
    print(f"questions={len(questions)} chunks={len(chunks)} -> {QUESTIONS}, qdrant:{COLLECTION}")


if __name__ == "__main__":
    main()
