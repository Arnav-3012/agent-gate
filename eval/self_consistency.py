"""Self-consistency signal for the 7B->14B cascade (D51).

Re-answers each question k times with the 7B at temperature>0 (same retrieval, same prompt as the
rag arm) and records `sc_agree` = mean F1 between each sample and the greedy answer already in the
baseline file. Low agreement = the 7B is unsure = escalate. Cost/latency of the k extra samples is
added to the row's cost/latency so cascade_cv charges the signal honestly. Resumable.

    python -m eval.self_consistency eval/runs/baseline_general.json --out eval/runs/baseline_general_sc.json
    python -m eval.cascade_cv eval/runs/baseline_general_sc.json eval/runs/baseline_general_14b.json \\
        --domain general --cheap default/rag --strong Qwen2.5-14B-Instruct-4bit/rag --signal sc_agree
"""
import argparse
import json
from pathlib import Path

from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Retriever
from engine.rag.single_pass import build_messages
from engine.tracing.schema import LLMStats
from eval.metrics import f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="baseline_general.json (7B rag rows are augmented)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="first N rag rows only (smoke test)")
    a = ap.parse_args()

    data = json.loads(a.baseline.read_text())
    if a.out.exists():
        data = json.loads(a.out.read_text())  # resume: rows already carrying sc_agree are skipped
    qs = {json.loads(l)["id"]: json.loads(l)
          for l in Path("eval/data/questions.jsonl").read_text().splitlines()}
    retriever, llm, pricing = Retriever.from_config(finance=False), LLMClient(), Pricing.from_yaml()
    ledger = Ledger(pricing, llm.model)
    rows = [r for r in data["results"] if r["mode"] == "rag"][: a.limit]
    for n, r in enumerate(rows, 1):
        if "sc_agree" in r:
            continue
        q = qs[r["id"]]
        msgs = build_messages(q["question"], retriever.search(q["question"]))
        calls = [llm.chat(msgs, max_tokens=64, temperature=a.temperature) for _ in range(a.k)]
        r["sc_agree"] = sum(f1(c.text.strip(), r["pred"]) for c in calls) / a.k
        r["cost"] += sum(ledger.price(LLMStats(prompt_tokens=c.prompt_tokens, completion_tokens=c.completion_tokens,
                                               cached_tokens=c.cached_tokens, latency_s=c.latency_s)) for c in calls)
        r["latency"] += sum(c.latency_s for c in calls)
        r["tokens"] += sum(c.total_tokens for c in calls)
        a.out.write_text(json.dumps(data, indent=1))
        print(f"[{n}/{len(rows)}] agree={r['sc_agree']:.2f} f1={r['f1']:.2f}", flush=True)
    a.out.write_text(json.dumps(data, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
