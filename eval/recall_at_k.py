"""Fix test T1 (D43): retrieval recall with no LLM. recall@k = share of questions whose gold chunks
are all in the top-k. Compares dense (baseline) with the company+year scope, at k=5 and 8, on the
same questions smoke_eval used. Writes eval/runs/fixtest_recall[_domain].json.
"""
import argparse
import json
from pathlib import Path

from engine.rag.retriever import Embedder, Retriever, point_id
from engine.rag.scope import parse_scope, title_parts
from scripts.smoke_eval import sample

CONFIGS = {"dense@5": (False, 5), "dense@8": (False, 8), "filter@5": (True, 5), "filter@8": (True, 8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["general", "finance"], default="finance")
    ap.add_argument("--per-type", type=int, default=20, help="finance: 20 (=60 Qs); general: 10 (=30 Qs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fin = a.domain == "finance"

    rows = [json.loads(l) for l in Path(
        f"eval/data/{'finance_questions' if fin else 'questions'}.jsonl").read_text().splitlines()]
    qs = sample(rows, a.per_type, a.seed, stable=fin)
    emb = Embedder()
    coll = "finance" if fin else "passages"
    dense = Retriever(emb, collection=coll)
    scoped = Retriever(emb, client=dense.client, collection=coll, metadata_filter=True)
    retrievers = {name: (scoped if filt else dense, k) for name, (filt, k) in CONFIGS.items()
                  if fin or not filt}  # the scope only exists for filing titles
    names = scoped._company_names() if fin else []

    out_rows = []
    for q in qs:
        gold = set(q["gold_chunk_ids"])
        row = {"id": q["id"], "type": q["type"], "hit": {}, "any": {}}
        for name, (r, k) in retrievers.items():
            hits = r.search(q["question"], top_k=k)
            got = {h.chunk.chunk_id for h in hits}
            row["hit"][name], row["any"][name] = gold <= got, bool(gold & got)
        if fin:
            row["gold_titles"] = sorted({p.payload["title"] for p in dense.client.retrieve(
                coll, [point_id(c) for c in gold], with_payload=["title"])})
            scope = parse_scope(q["question"], names)
            row["parsed"] = {"companies": sorted(scope.companies), "years": sorted(scope.years)}
            gold_cos = {p[0] for t in row["gold_titles"] if (p := title_parts(t))}
            row["company_ok"] = gold_cos <= scope.companies and bool(scope)
        out_rows.append(row)

    def rate(rs: list[dict], name: str) -> str:
        n = sum(r["hit"][name] for r in rs)
        return f"{n}/{len(rs)} = {n / len(rs):.0%}"

    print(f"{a.domain}: n={len(out_rows)}")
    for name in retrievers:
        n_any = sum(r["any"][name] for r in out_rows)
        print(f"  recall {name:<9} all-gold {rate(out_rows, name)}   any-gold {n_any}/{len(out_rows)} = {n_any / len(out_rows):.0%}")
        for typ in sorted({r['type'] for r in out_rows}):
            print(f"    {typ:<18} {rate([r for r in out_rows if r['type'] == typ], name)}")
    if fin:
        bad = [r for r in out_rows if not r["company_ok"]]
        print(f"company parsed correctly: {len(out_rows) - len(bad)}/{len(out_rows)}")
        for r in bad:
            print(f"  PARSE MISS {r['id']} parsed={r['parsed']['companies']} gold={r['gold_titles'][:2]}")
        lost = [r for r in out_rows if r["hit"]["dense@5"] and not r["hit"]["filter@5"]]
        print(f"filter lost a dense hit: {len(lost)}", *[r["id"] for r in lost])
    out = Path(f"eval/runs/fixtest_recall{'' if fin else '_general'}.json")
    out.write_text(json.dumps({"configs": list(retrievers), "rows": out_rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
