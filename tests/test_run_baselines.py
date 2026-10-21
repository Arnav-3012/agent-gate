import json
from pathlib import Path

import httpx

from engine.agent.loop import AgentConfig
from engine.cost.ledger import Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Chunk, Hit
from eval.metrics import f1
from eval.run_baselines import run_baselines

QS = [{"id": f"q{i}", "type": "bridge", "level": lvl, "question": "Who?", "answer": "Paris",
       "gold_chunk_ids": ["c1"]} for i, lvl in enumerate(["easy", "hard"])]


class FakeRetriever:
    collection, client = "x", None

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        return [Hit(chunk=Chunk(chunk_id="c1", title="T", text="Paris is the capital", source="s"), score=0.9)]


def make_llm() -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 2}})
    return LLMClient(base_url="http://x/v1", model="m", transport=httpx.MockTransport(handler))


def go(out: Path, resume: bool = False) -> list[dict]:
    return run_baselines(QS, ["rag"], out, "m", FakeRetriever(), make_llm, f1,
                         Pricing.from_yaml(), AgentConfig.from_yaml(), {}, resume)


def test_rows_have_all_fields_and_level(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    rows = go(out)
    assert [r["level"] for r in rows] == ["easy", "hard"]
    assert {"em", "f1", "tokens", "cost", "latency", "stop", "gold_found", "rag_gold"} <= set(rows[0])
    assert rows[0]["f1"] == 1.0 and rows[0]["gold_found"] and rows[0]["rag_gold"]
    assert rows[0]["top1"] == 0.9 and rows[0]["gap12"] == 0.9  # one hit: gap = its score
    assert json.loads(out.read_text())["model"] == "m"


def test_resume_skips_finished_pairs(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    go(out)
    calls = []
    FakeRetriever.search, orig = lambda s, q, top_k=None: calls.append(q) or [], FakeRetriever.search
    try:
        rows = go(out, resume=True)
    finally:
        FakeRetriever.search = orig
    assert calls == [] and len(rows) == 2


def test_second_arm_process_keeps_first_arms_rows(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    go(out)  # rag
    run_baselines(QS, ["agent"], out, "m", FakeRetriever(), make_llm, f1, Pricing.from_yaml(),
                  AgentConfig.from_yaml(), {}, resume=True)
    assert {r["mode"] for r in json.loads(out.read_text())["results"]} == {"rag", "agent"}


def test_resume_refuses_a_changed_config(tmp_path: Path) -> None:
    import pytest
    out = tmp_path / "b.json"
    args = (QS, ["rag"], out, "m", FakeRetriever(), make_llm, f1, Pricing.from_yaml(), AgentConfig.from_yaml(), {})
    run_baselines(*args, meta={"top_k": 5})
    with pytest.raises(SystemExit):
        run_baselines(*args, resume=True, meta={"top_k": 8})
