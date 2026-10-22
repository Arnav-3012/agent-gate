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
import os
from pathlib import Path

import httpx
import yaml

from engine.llm.client import LLMCall, LLMClient, Message
from eval.metrics import finance_f1, is_numeric_gold

SYSTEM = (
    "You grade answers to financial questions against a reference answer. Judge meaning, not "
    "wording: extra justification is fine; a different figure, entity or conclusion is wrong. "
    "Reply with exactly one word: correct, partial, or incorrect."
)
SYSTEM_GENERAL = (
    "You grade answers to short factual questions (HotpotQA/SQuAD style) against a reference answer. "
    "Judge meaning, not wording: aliases, date/number formats and extra words are fine; a different "
    "entity, number or yes/no is wrong. Reply with exactly one word: correct, partial, or incorrect."
)
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class HostedJudge:
    """Anthropic Messages API judge (D42): a different family from the Qwen answerers, so no
    self-preference. Same `.chat` shape as LLMClient; tracks tokens for the cost report."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None, max_usd: float = 0.50) -> None:
        self.max_usd = max_usd  # hard stop: a bug or a huge file must not run up the bill
        self.model = model or os.getenv("JUDGE_MODEL", "claude-haiku-5-5")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("hosted judge needs ANTHROPIC_API_KEY in .env (never commit it)")
        self._http = httpx.Client(timeout=60, transport=transport, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        self.tokens_in = self.tokens_out = 0

    def chat(self, messages: list[Message], max_tokens: int = 8) -> LLMCall:
        if self.cost_usd() >= self.max_usd:
            raise SystemExit(f"hosted judge stopped: ~${self.cost_usd():.4f} >= --max-usd {self.max_usd}")
        system = "\n".join(m.content for m in messages if m.role == "system")
        body = {"model": self.model, "max_tokens": max_tokens, "system": system,
                "messages": [m.model_dump() for m in messages if m.role != "system"]}
        resp = self._http.post(self.URL, json=body)
        resp.raise_for_status()
        out = resp.json()
        self.tokens_in += out["usage"]["input_tokens"]
        self.tokens_out += out["usage"]["output_tokens"]
        return LLMCall(text="".join(b["text"] for b in out["content"] if b["type"] == "text"),
                       prompt_tokens=out["usage"]["input_tokens"],
                       completion_tokens=out["usage"]["output_tokens"], latency_s=0.0)

    def cost_usd(self, pricing_path: Path = Path("config/pricing.yaml")) -> float:
        """At the verified reference rate (the judge IS the reference model, claude-haiku-5-5)."""
        rate = yaml.safe_load(pricing_path.read_text())["usd_per_1m_tokens"]
        return (self.tokens_in * rate["input"] + self.tokens_out * rate["output"]) / 1e6


def parse_verdict(text: str) -> str | None:
    """First verdict word in the reply; 'incorrect' is checked before 'correct' (substring)."""
    t = text.lower()
    hits = [(t.find(w), w) for w in ("incorrect", "partial", "correct") if w in t]
    return min(hits)[1] if hits else None


def judge_row(llm: LLMClient | HostedJudge, question: str, gold: str, pred: str, system: str = SYSTEM) -> str:
    call = llm.chat([
        Message(role="system", content=system),
        Message(role="user",
                content=f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}"),
    ], max_tokens=8)
    return parse_verdict(call.text) or "incorrect"  # unparseable reply is not credit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_file", type=Path)
    ap.add_argument("--domain", choices=["finance", "general"], default="finance",
                    help="general = short-answer prompt, every row judged (no numeric shortcut)")
    ap.add_argument("--questions", type=Path, default=None,
                    help="default eval/data/finance_questions.jsonl or questions.jsonl by --domain")
    ap.add_argument("--judge", choices=["local", "hosted"], default="local",
                    help="hosted = Anthropic API (D42, needs ANTHROPIC_API_KEY); local = OpenAI-compatible server")
    ap.add_argument("--max-usd", type=float, default=0.50, help="hosted judge: abort past this estimated spend")
    ap.add_argument("--model", default=None, help="judge model id (local: LLM_MODEL env; hosted: JUDGE_MODEL env)")
    ap.add_argument("--base-url", default=None)
    a = ap.parse_args()

    data = json.loads(a.result_file.read_text())
    fin = a.domain == "finance"
    qpath = a.questions or Path(f"eval/data/{'finance_questions' if fin else 'questions'}.jsonl")
    questions = {q["id"]: q["question"] for q in map(json.loads, qpath.read_text().splitlines())}
    llm = HostedJudge(a.model, max_usd=a.max_usd) if a.judge == "hosted" else LLMClient(base_url=a.base_url, model=a.model)
    answering = data.get("model") or "default"
    for i, r in enumerate(data["results"], 1):
        if fin and is_numeric_gold(r["gold"]):
            r["score"], r["verdict"] = finance_f1(r["pred"], r["gold"]), "numeric"
        else:
            r["verdict"] = judge_row(llm, questions[r["id"]], r["gold"], r["pred"], SYSTEM if fin else SYSTEM_GENERAL) if r["pred"] else "incorrect"
            r["score"] = SCORE[r["verdict"]]
        r["self_judged"] = llm.model == answering
        print(f"[{i}/{len(data['results'])}] {r['mode']:<6} {r['verdict']}", flush=True)
    data["judge_model"] = llm.model
    if isinstance(llm, HostedJudge):
        print(f"judge tokens in/out {llm.tokens_in}/{llm.tokens_out}, ~${llm.cost_usd():.4f}")
    out = a.result_file.with_suffix(".judged.json")
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
