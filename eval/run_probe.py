"""Router probe set (D31, parallel-work A4): ~40 labelled questions, each with a category and a
hypothesis `expected_arm` (the cheapest arm we think suffices). A wrong hypothesis is a finding.

    python eval/run_probe.py draw                     # add 5 questions per category from the datasets
    python eval/run_probe.py run --domain general --modes rag --tag 7b --model ... --base-url ...
    python eval/run_probe.py report --domain general eval/runs/probe/*.json

Numeric finance questions (`metrics-generated`) are never drawn (D39).
Hand-written categories (not-in-corpus, ambiguous) have no gold answer; they are scored by whether
the arm abstained or asked for clarification, not by F1. Results go to eval/runs/probe/ so
benchmark.py's eval/runs/*.json glob never mixes probe rows into the headline tables.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from engine.router.arms import ArmRegistry

PROBE = Path("eval/probe_set.jsonl")
ABSTAIN_CATEGORIES = {"not-in-corpus", "ambiguous"}
ABSTAIN = re.compile(
    r"don't know|do not know|not (?:mentioned|provided|found|available|specified|enough)|cannot|can't|"
    r"unable|unclear|unanswerable|no information|insufficient|clarif|which .{0,40} do you mean|please specify",
    re.IGNORECASE)
# (domain, dataset type) -> (probe category, hypothesised cheapest sufficient arm); Arnav edits these.
DRAW: dict[tuple[str, str], tuple[str, str]] = {
    ("general", "single"): ("single-fact", "qwen7b-rag"),
    ("general", "comparison"): ("comparison", "qwen7b-rag"),
    ("general", "bridge"): ("bridge", "qwen14b-rag"),
    ("finance", "domain-relevant"): ("finance-text", "qwen14b-rag"),
    ("finance", "novel-generated"): ("finance-reasoning", "qwen14b-agent"),
}
NUMERIC_TYPES = {"metrics-generated"}  # D39: numeric finance questions are not run in Phase 3, so never drawn


class ProbeQuestion(BaseModel):
    id: str
    domain: Literal["general", "finance"]
    category: str
    answer_type: Literal["span", "number", "sentence", "yes_no", "abstain"]
    expected_arm: str
    question: str
    answer: str
    gold_chunk_ids: list[str] = []

    def as_eval_row(self) -> dict:
        """The row shape run_baselines expects; `type` carries the category."""
        return {**self.model_dump(), "type": self.category, "level": "probe"}


def load_probe(path: Path = PROBE, arms: ArmRegistry | None = None) -> list[ProbeQuestion]:
    qs = [ProbeQuestion.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]
    if len({q.id for q in qs}) != len(qs):
        raise ValueError("duplicate probe ids")
    if arms:
        known = {a.name for a in arms.arms}
        if bad := {q.expected_arm for q in qs} - known:
            raise ValueError(f"expected_arm not in registry: {sorted(bad)}")
    return qs


def abstained(pred: str) -> bool:
    return bool(ABSTAIN.search(pred))


def probe_score(category: str, answer_type: str, row: dict) -> float:
    """Abstain categories: 1 if the arm declined/asked, else 0. Others: the stored F1/judge score."""
    if category in ABSTAIN_CATEGORIES or answer_type == "abstain":
        return float(abstained(row["pred"]))
    return row.get("score", row["f1"])


def report(qs: list[ProbeQuestion], arms: dict[str, list[dict]], costs_tiebreak: bool = True) -> dict:
    """{category: {arm: {score, tokens, cost, n}, 'winner': arm, 'expected': arm, 'match': bool}}.
    The winner is the highest-scoring arm, ties broken by lower mean cost."""
    meta = {q.id: q for q in qs}
    out: dict = {}
    for cat in sorted({q.category for q in qs}):
        per: dict = {}
        for arm, rows in arms.items():
            rs = [r for r in rows if r["id"] in meta and meta[r["id"]].category == cat]
            if rs:
                per[arm] = {"score": mean(probe_score(cat, meta[r["id"]].answer_type, r) for r in rs),
                            "tokens": mean(r["tokens"] for r in rs), "cost": mean(r["cost"] for r in rs), "n": len(rs)}
        if not per:
            continue
        win = min(per, key=lambda a: (-per[a]["score"], per[a]["cost"] if costs_tiebreak else 0))
        expected = next(q.expected_arm for q in qs if q.category == cat)
        out[cat] = {**per, "winner": win, "expected": expected, "match": win == expected}
    return out


def draw(per_cat: int = 5, seed: int = 0, out: Path = PROBE) -> int:
    """Append per_cat dataset questions per category with the hypothesis arm from DRAW (skips ids present)."""
    if bad := {typ for _, typ in DRAW} & NUMERIC_TYPES:  # check before writing anything
        raise ValueError(f"{sorted(bad)} excluded from Phase 3 runs (D39)")
    have = {q.id for q in load_probe(out)} if out.exists() else set()
    rng = random.Random(seed)
    added = 0
    with out.open("a") as f:
        for (domain, typ), (cat, arm) in DRAW.items():
            path = Path("eval/data/finance_questions.jsonl" if domain == "finance" else "eval/data/questions.jsonl")
            pool = [r for r in map(json.loads, path.read_text().splitlines()) if r["type"] == typ and r["id"] not in have]
            for r in rng.sample(pool, per_cat):
                atype = "sentence" if domain == "finance" else "span"
                q = ProbeQuestion(id=r["id"], domain=domain, category=cat, answer_type=atype, expected_arm=arm,
                                  question=r["question"], answer=r["answer"], gold_chunk_ids=r["gold_chunk_ids"])
                f.write(q.model_dump_json() + "\n")
                added += 1
    return added


def _print(rep: dict) -> None:
    for cat, e in rep.items():
        arms = {k: v for k, v in e.items() if isinstance(v, dict)}
        print(f"\n{cat}: winner {e['winner']}, hypothesis {e['expected']} -> {'MATCH' if e['match'] else 'MISS'}")
        for a, s in arms.items():
            print(f"  {a:<24} score {s['score']:.2f}  {s['tokens']:>6.0f} tok  ${s['cost'] * 1000:.4f}/1k  n={s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draw", "run", "report"])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--domain", choices=["general", "finance"], default="general")
    ap.add_argument("--modes", default="rag,agent")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "draw":
        print(f"added {draw()} questions to {PROBE}")
        return
    qs = [q for q in load_probe(PROBE, ArmRegistry.from_yaml()) if q.domain == a.domain]
    if a.cmd == "run":
        from engine.agent.loop import AgentConfig
        from engine.cost.ledger import Pricing
        from engine.llm.client import LLMClient
        from engine.rag.retriever import Retriever
        from engine.rag.single_pass import FINANCE_SYSTEM
        from eval.metrics import f1, finance_f1
        from eval.run_baselines import run_baselines

        fin = a.domain == "finance"
        cfg, kw = AgentConfig.from_yaml(), {}
        if fin:
            kw = {"max_answer_tokens": 192, "system": FINANCE_SYSTEM}
            cfg = cfg.model_copy(update={"max_answer_tokens": 192, "answer_system": FINANCE_SYSTEM})
        out = Path(f"eval/runs/probe/probe_{'_'.join(t for t in (a.domain, a.tag) if t)}.json")
        run_baselines([q.as_eval_row() for q in qs], a.modes.split(","), out, a.model,
                      Retriever.from_config(finance=fin),
                      lambda: LLMClient(base_url=a.base_url, model=a.model),
                      finance_f1 if fin else f1, Pricing.from_yaml(), cfg, kw, a.resume)
        print(f"wrote {out}")
        return
    arms: dict[str, list[dict]] = defaultdict(list)
    registry = ArmRegistry.from_yaml()
    names = {f"{x.model.split('/')[-1]}/{x.mode}": x.name for x in registry.arms}
    for p in a.files:
        d = json.loads(p.read_text())
        model = (d.get("model") or "default").split("/")[-1]
        for r in d["results"]:
            key = f"{model}/{r['mode']}"
            arms[names.get(key, key)].append(r)
    _print(report(qs, arms))


if __name__ == "__main__":
    main()
