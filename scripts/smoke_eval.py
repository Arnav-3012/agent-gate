"""Phase 2 go/no-go: 30 Qs (10 bridge, 10 comparison, 10 squad) through RAG and agent.

Prints EM/F1/tokens/cost/latency per mode x type, the same split by whether RAG's top-k already
held every gold paragraph (the agent can only help where it did not), and a verdict on whether the
agent beats RAG on multi-hop (bridge) by --margin F1. Writes eval/runs/smoke_eval[_tag].json.
"""
import argparse
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean

from engine.agent.loop import AgentConfig, run_agent
from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Chunk, Hit, Retriever, point_id
from engine.rag.single_pass import FINANCE_SYSTEM, run_rag
from engine.tracing.emitter import Emitter
from eval.metrics import exact_match, f1, finance_f1


def sample(rows: list[dict], per_type: int, seed: int, stable: bool = False) -> list[dict]:
    """Stratified by type. stable=True takes the first k of one seeded permutation per type, so a
    larger per_type extends a smaller run (needed for --resume growth, e.g. 20 -> 50 per type).
    The default keeps the original sampler so the existing general-set result files stay valid."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    if stable:
        return [q for t in sorted(by_type) for q in rng.sample(by_type[t], len(by_type[t]))[:per_type]]
    return [q for t in sorted(by_type) for q in rng.sample(by_type[t], per_type)]


def summarize(results: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{mode: {type: mean metrics}} plus an 'all' bucket per mode."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["mode"], r["type"])].append(r)
        groups[(r["mode"], "all")].append(r)
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (mode, typ), rs in groups.items():
        out[mode][typ] = {k: mean(r[k] for r in rs) for k in ("em", "f1", "tokens", "cost", "latency")}
        out[mode][typ]["n"] = len(rs)
    return out


def by_retrieval(results: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{mode: {'rag_hit'|'rag_miss': mean f1/tokens/n}} split on RAG's gold recall for that question."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for mode in sorted({r["mode"] for r in results}):
        for flag, name in ((True, "rag_hit"), (False, "rag_miss")):
            rs = [r for r in results if r["mode"] == mode and r["rag_gold"] == flag]
            if rs:
                out[mode][name] = {"f1": mean(r["f1"] for r in rs),
                                   "tokens": mean(r["tokens"] for r in rs), "n": len(rs)}
    return out


class OracleRetriever:
    """Diagnostic arm: returns the question's gold chunks, so the LLM's reading is tested alone."""

    def __init__(self, base: Retriever) -> None:
        self.base, self.gold_ids = base, []

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        pts = self.base.client.retrieve(self.base.collection, [point_id(c) for c in self.gold_ids],
                                        with_payload=True)
        return [Hit(chunk=Chunk(**p.payload), score=1.0) for p in pts]


def verdict(summary: dict, margin: float) -> tuple[bool, str]:
    if "bridge" not in summary.get("rag", {}) or "agent" not in summary:
        return False, "no verdict (needs rag+agent on bridge questions)"
    rag, agent = summary["rag"]["bridge"], summary["agent"]["bridge"]
    gain = agent["f1"] - rag["f1"]
    ok = gain >= margin
    return ok, (f"{'GO' if ok else 'NO-GO'}: bridge F1 agent {agent['f1']:.3f} vs rag {rag['f1']:.3f} "
                f"(gain {gain:+.3f}, need >= {margin}); cost x{agent['cost'] / max(rag['cost'], 1e-12):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--model", default=None, help="LLM model id (default: LLM_MODEL env)")
    ap.add_argument("--base-url", default=None, help="LLM server (default: LLM_BASE_URL env)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. 14b")
    ap.add_argument("--domain", choices=["general", "finance"], default="general",
                    help="finance: finance_questions.jsonl + 'finance' collection + numeric-tolerant F1")
    ap.add_argument("--exclude-types", default="",
                    help="comma list of question types to skip AFTER sampling (so the other types keep the "
                         "exact questions of earlier runs), e.g. metrics-generated")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the existing output file: skip (question, mode) pairs already done")
    ap.add_argument("--modes", default="rag,agent",
                    help="comma list from rag,agent,oracle (oracle = gold chunks as context; "
                         "separates retrieval failures from reading failures)")
    a = ap.parse_args()
    modes = a.modes.split(",")
    tag = "_".join(t for t in (a.domain if a.domain != "general" else "", a.tag) if t)
    out = Path(f"eval/runs/smoke_eval{'_' + tag if tag else ''}.json")

    fin = a.domain == "finance"
    rows = [json.loads(l) for l in
            Path("eval/data/finance_questions.jsonl" if fin else "eval/data/questions.jsonl")
            .read_text().splitlines()]
    skip = set(filter(None, a.exclude_types.split(",")))
    qs = [q for q in sample(rows, a.per_type, a.seed, stable=fin) if q["type"] not in skip]
    score = finance_f1 if fin else f1
    retriever = Retriever(collection="finance") if fin else Retriever()
    pricing, cfg = Pricing.from_yaml(), AgentConfig.from_yaml()
    answer_kw = {"max_answer_tokens": 192, "system": FINANCE_SYSTEM} if fin else {}
    if fin:  # sentence-length gold answers: lift the short-answer prompt and token cap on every arm
        cfg = cfg.model_copy(update={"max_answer_tokens": 192, "answer_system": FINANCE_SYSTEM})
    oracle = OracleRetriever(retriever)
    results: list[dict] = json.loads(out.read_text())["results"] if a.resume and out.exists() else []
    done = {(r["id"], r["mode"]) for r in results}
    if done:
        print(f"resuming: {len(done)} (question, mode) pairs already in {out}")

    def save(summary: dict | None = None, split: dict | None = None, line: str = "") -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": a.model, "summary": summary, "by_retrieval": split,
                                   "verdict": line, "results": results}, indent=1))

    for i, q in enumerate(qs, 1):
        oracle.gold_ids = q["gold_chunk_ids"]
        todo = [m for m in modes if (q["id"], m) not in done]
        if not todo:
            continue
        # RAG's own top-k, independent of which arms run in this process (arms may run separately)
        rag_gold = set(q["gold_chunk_ids"]) <= {h.chunk.chunk_id for h in retriever.search(q["question"])}
        for mode in todo:
            llm = LLMClient(base_url=a.base_url, model=a.model)
            em = Emitter(uuid.uuid4().hex, [], Ledger(pricing, llm.model))
            pred = (run_agent(q["question"], retriever, llm, em, cfg) if mode == "agent"
                    else run_rag(q["question"], oracle if mode == "oracle" else retriever, llm, em,
                                 **answer_kw))
            stop = em.events[-1]
            seen = {h.chunk_id for e in em.events if e.type == "retrieve" for h in e.hits}
            results.append({"id": q["id"], "type": q["type"], "mode": mode, "pred": pred,
                            "gold": q["answer"], "em": exact_match(pred, q["answer"]),
                            "f1": score(pred, q["answer"]), "tokens": stop.total_tokens,
                            "cost": stop.total_cost_usd, "latency": stop.total_latency_s,
                            "stop": stop.reason, "steps": stop.steps, "error": stop.error,
                            "gold_found": set(q["gold_chunk_ids"]) <= seen, "rag_gold": rag_gold})
        print(f"[{i}/{len(qs)}] {q['type']:<18} " + " ".join(
            f"{r['mode']} f1={r['f1']:.2f} ({r['tokens']} tok)" for r in results[-len(todo):]), flush=True)
        save()  # a crash later never loses finished questions

    results = [r for r in results if r["mode"] in modes and r["id"] in {q["id"] for q in qs}]
    summary = summarize(results)
    print(f"\n{'mode':<6} {'type':<18} {'n':>3} {'EM':>6} {'F1':>6} {'tok':>7} {'$/q':>11} {'lat s':>6}")
    for mode in modes:
        for typ in sorted(summary[mode], key=lambda t: (t == "all", t)):
            s = summary[mode][typ]
            print(f"{mode:<6} {typ:<18} {s['n']:>3.0f} {s['em']:>6.2f} {s['f1']:>6.3f} "
                  f"{s['tokens']:>7.0f} {s['cost']:>11.7f} {s['latency']:>6.1f}")
    split = by_retrieval(results)
    print("\nsplit by RAG gold recall (agent can only help on rag_miss):")
    for name in ("rag_hit", "rag_miss"):
        if name in split.get("rag", {}):
            print(f"  {name:<9} n={split['rag'][name]['n']:.0f}  " + "  ".join(
                f"{m} F1 {split[m][name]['f1']:.3f}" for m in ("rag", "agent", "oracle")
                if name in split.get(m, {})))
    errs = [r for r in results if r["error"]]
    print(f"errors: {len(errs)}", *[f"{r['id']}: {r['error'][:80]}" for r in errs], sep="\n  ")
    _, line = verdict(summary, a.margin)
    print("\n" + line)
    save(summary, split, line)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
