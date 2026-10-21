"""Phase 3 baseline runner: one (domain, model) over a stratified question set, any arms, resumable.

Generalises scripts/smoke_eval.py: same sampler, arms and scoring, plus `level` on every row so
benchmark.py can report per-difficulty tables. Writes eval/runs/baseline_<domain>[_tag].json with
the same {"model", "results"} shape pareto.py reads. Run one arm per process on a 24GB machine.
"""
import argparse
import json
import uuid
from collections.abc import Callable
from pathlib import Path

from engine.agent.loop import AgentConfig, run_agent
from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Retriever
from engine.rag.single_pass import FINANCE_SYSTEM, run_rag
from engine.tracing.emitter import Emitter
from eval.metrics import exact_match, f1, finance_f1
from scripts.smoke_eval import OracleRetriever, sample

MODES = ("rag", "agent", "oracle")


def run_question(q: dict, mode: str, retriever, oracle: OracleRetriever, make_llm: Callable[[], LLMClient],
                 pricing: Pricing, cfg: AgentConfig, score: Callable[[str, str], float],
                 answer_kw: dict, rag_gold: bool) -> dict:
    """One (question, arm) -> one result row. A backend failure becomes a row with `error` set."""
    llm = make_llm()
    em = Emitter(uuid.uuid4().hex, [], Ledger(pricing, llm.model))
    if mode == "agent":
        pred = run_agent(q["question"], retriever, llm, em, cfg)
    else:
        pred = run_rag(q["question"], oracle if mode == "oracle" else retriever, llm, em, **answer_kw)
    stop = em.events[-1]
    seen = {h.chunk_id for e in em.events if e.type == "retrieve" for h in e.hits}
    return {"id": q["id"], "type": q["type"], "level": q.get("level", "unknown"), "mode": mode,
            "pred": pred, "gold": q["answer"], "em": exact_match(pred, q["answer"]),
            "f1": score(pred, q["answer"]), "tokens": stop.total_tokens, "cost": stop.total_cost_usd,
            "latency": stop.total_latency_s, "stop": stop.reason, "steps": stop.steps,
            "error": stop.error, "gold_found": set(q["gold_chunk_ids"]) <= seen, "rag_gold": rag_gold}


def run_baselines(qs: list[dict], modes: list[str], out: Path, model: str | None, retriever,
                  make_llm: Callable[[], LLMClient], score: Callable[[str, str], float],
                  pricing: Pricing, cfg: AgentConfig, answer_kw: dict, resume: bool = False) -> list[dict]:
    """Runs every missing (question, mode) pair, saving after each question so a crash loses nothing."""
    results: list[dict] = json.loads(out.read_text())["results"] if resume and out.exists() else []
    done = {(r["id"], r["mode"]) for r in results}
    oracle = OracleRetriever(retriever)
    ids = {q["id"] for q in qs}

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": model, "results": results}, indent=1))

    for i, q in enumerate(qs, 1):
        todo = [m for m in modes if (q["id"], m) not in done]
        if not todo:
            continue
        oracle.gold_ids = q["gold_chunk_ids"]
        rag_gold = set(q["gold_chunk_ids"]) <= {h.chunk.chunk_id for h in retriever.search(q["question"])}
        for m in todo:
            results.append(run_question(q, m, retriever, oracle, make_llm, pricing, cfg, score,
                                        answer_kw, rag_gold))
        print(f"[{i}/{len(qs)}] {q['type']:<18} " + " ".join(
            f"{r['mode']} f1={r['f1']:.2f} ({r['tokens']} tok)" for r in results[-len(todo):]), flush=True)
        save()
    results = [r for r in results if r["mode"] in modes and r["id"] in ids]
    save()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["general", "finance"], default="general")
    ap.add_argument("--modes", default="rag,agent", help=f"comma list from {','.join(MODES)}")
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="LLM model id (default: LLM_MODEL env)")
    ap.add_argument("--base-url", default=None, help="LLM server (default: LLM_BASE_URL env)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. 14b")
    ap.add_argument("--resume", action="store_true", help="skip (question, mode) pairs already in the file")
    ap.add_argument("--exclude-types", default=None,
                    help="comma list of question types to skip AFTER sampling (keeps other types' questions "
                         "identical to earlier runs). Default: metrics-generated for --domain finance (D39, "
                         "numeric questions are not run in Phase 3); pass '' to include them")
    a = ap.parse_args()
    modes = a.modes.split(",")
    if bad := set(modes) - set(MODES):
        ap.error(f"unknown modes {sorted(bad)}")

    fin = a.domain == "finance"
    rows = [json.loads(l) for l in Path(
        f"eval/data/{'finance_questions' if fin else 'questions'}.jsonl").read_text().splitlines()]
    exclude = a.exclude_types if a.exclude_types is not None else ("metrics-generated" if a.domain == "finance" else "")
    skip = set(filter(None, exclude.split(",")))
    qs = [q for q in sample(rows, a.per_type, a.seed, stable=True) if q["type"] not in skip]
    cfg = AgentConfig.from_yaml()
    answer_kw: dict = {}
    if fin:  # sentence-length gold answers: lift the short-answer prompt and token cap on every arm
        answer_kw = {"max_answer_tokens": 192, "system": FINANCE_SYSTEM}
        cfg = cfg.model_copy(update={"max_answer_tokens": 192, "answer_system": FINANCE_SYSTEM})
    tag = "_".join(t for t in (a.domain, a.tag) if t)
    out = Path(f"eval/runs/baseline_{tag}.json")
    run_baselines(qs, modes, out, a.model, Retriever(collection="finance") if fin else Retriever(),
                  lambda: LLMClient(base_url=a.base_url, model=a.model),
                  finance_f1 if fin else f1, Pricing.from_yaml(), cfg, answer_kw, a.resume)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
