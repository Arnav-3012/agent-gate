"""Offline LLM-judge rescoring of saved eval result files (never reruns an arm).

For each row whose gold answer is NOT a short number, a reference-guided judge labels the
prediction correct / partially correct / incorrect (1 / 0.5 / 0). Numeric-gold rows keep the
deterministic `finance_f1`. Writes <input>.judged.json with `score` per row (originals untouched)
and records the judge model; rows where judge == answering model are flagged `self_judged`
because LLM judges favour their own outputs. Validate against ~30 hand-checked rows before
trusting it.

    python eval/judge.py eval/runs/smoke_eval_finance.json --questions eval/data/finance_questions.jsonl
"""
import argparse
import json
from pathlib import Path

from engine.llm.client import LLMClient, Message
from eval.metrics import finance_f1, is_numeric_gold

SYSTEM = (
    "You grade answers to financial questions against a reference answer. Judge meaning, not "
    "wording: extra justification is fine; a different figure, entity or conclusion is wrong. "
    "Reply with exactly one word: correct, partial, or incorrect."
)
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def parse_verdict(text: str) -> str | None:
    """First verdict word in the reply; 'incorrect' is checked before 'correct' (substring)."""
    t = text.lower()
    hits = [(t.find(w), w) for w in ("incorrect", "partial", "correct") if w in t]
    return min(hits)[1] if hits else None


def judge_row(llm: LLMClient, question: str, gold: str, pred: str) -> str:
    call = llm.chat([
        Message(role="system", content=SYSTEM),
        Message(role="user",
                content=f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}"),
    ], max_tokens=8)
    return parse_verdict(call.text) or "incorrect"  # unparseable reply is not credit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_file", type=Path)
    ap.add_argument("--questions", type=Path, default=Path("eval/data/finance_questions.jsonl"))
    ap.add_argument("--model", default=None, help="judge model id (default: LLM_MODEL env)")
    ap.add_argument("--base-url", default=None)
    a = ap.parse_args()

    data = json.loads(a.result_file.read_text())
    questions = {q["id"]: q["question"] for q in map(json.loads, a.questions.read_text().splitlines())}
    llm = LLMClient(base_url=a.base_url, model=a.model)
    answering = data.get("model") or "default"
    for i, r in enumerate(data["results"], 1):
        if is_numeric_gold(r["gold"]):
            r["score"], r["verdict"] = finance_f1(r["pred"], r["gold"]), "numeric"
        else:
            r["verdict"] = judge_row(llm, questions[r["id"]], r["gold"], r["pred"]) if r["pred"] else "incorrect"
            r["score"] = SCORE[r["verdict"]]
        r["self_judged"] = llm.model == answering
        print(f"[{i}/{len(data['results'])}] {r['mode']:<6} {r['verdict']}", flush=True)
    data["judge_model"] = llm.model
    out = a.result_file.with_suffix(".judged.json")
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
