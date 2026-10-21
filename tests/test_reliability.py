import json
from pathlib import Path

import httpx
import pytest

from engine.api.health import Health, decide_mode
from engine.router.arms import ArmRegistry
from engine.tracing.schema import Stop, event_adapter
from eval.run_probe import ProbeQuestion, abstained, load_probe, probe_score, report
from scripts.drill import DRILLS, plan
from scripts.record_replay import record
from scripts.warmup import warmup

UP, DOWN = Health(ok=True, llm=True, qdrant=True), Health(ok=False, llm=False, qdrant=True)


@pytest.mark.parametrize("health,replay,expected", [
    (UP, True, "live"), (UP, False, "live"),
    (DOWN, True, "replay"), (Health(ok=False, llm=True, qdrant=False), True, "replay"),
    (None, True, "replay"), (None, False, "unavailable"), (DOWN, False, "unavailable"),
    (Health(ok=True, llm=True, qdrant=False), True, "replay"),  # inconsistent payload: trust the parts
])
def test_decide_mode(health, replay, expected) -> None:
    assert decide_mode(health, replay) == expected


def fake_engine(error: str | None = None) -> httpx.Client:
    stop = Stop(trace_id="a" * 32, seq=0, ts=1.0, reason="rag_done", steps=1, total_tokens=5,
                total_cost_usd=0.0, total_latency_s=0.1, error=error)
    event = json.loads(event_adapter.dump_json(stop))

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/run":
            return httpx.Response(202, json={"trace_id": "a" * 32})
        if req.url.path == "/health":
            return httpx.Response(200, json=UP.model_dump())
        return httpx.Response(200, json={"trace_id": "a" * 32, "events": [event], "done": True})

    return httpx.Client(base_url="http://e", transport=httpx.MockTransport(handler))


def test_record_replay_writes_valid_traces(tmp_path: Path) -> None:
    paths = record(fake_engine(), ["Who directed Inception?", "Hi?"], tmp_path / "r", arm="qwen7b-rag", poll_s=0)
    assert [p.name for p in paths] == ["01-who-directed-inception.jsonl", "02-hi.jsonl"]
    assert event_adapter.validate_json(paths[0].read_text().splitlines()[0]).type == "stop"


def test_warmup_all_arms_and_errors() -> None:
    res = warmup(fake_engine(), ["a", "b"])
    assert res == {"_health": "live", "a": "ok", "b": "ok"}
    assert warmup(fake_engine(error="boom"), ["a"])["a"] == "error: boom"


def test_drill_plan_lists_every_scenario() -> None:
    assert len(plan()) == len(DRILLS) == 2 and all("replay" in l for l in plan())


def test_probe_set_loads_and_expected_arms_exist() -> None:
    qs = load_probe(Path("eval/probe_set.jsonl"), ArmRegistry.from_yaml())
    assert {q.category for q in qs} >= {"not-in-corpus", "ambiguous"}
    assert q_row(qs[0])["type"] == qs[0].category


def q_row(q: ProbeQuestion) -> dict:
    return q.as_eval_row()


def test_probe_validation_rejects_bad_arm(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"id": "x", "domain": "general", "category": "c", "answer_type": "span",
                             "expected_arm": "nope", "question": "q", "answer": "a"}))
    with pytest.raises(ValueError):
        load_probe(p, ArmRegistry.from_yaml())


def test_abstention_scoring_and_report_winner() -> None:
    assert abstained("I don't know.") and abstained("The context does not provide enough") is False
    assert abstained("This is not mentioned in the context")
    qs = [ProbeQuestion(id="1", domain="general", category="not-in-corpus", answer_type="abstain",
                        expected_arm="qwen7b-rag", question="q", answer="x"),
          ProbeQuestion(id="2", domain="general", category="single-fact", answer_type="span",
                        expected_arm="qwen7b-rag", question="q", answer="Paris")]

    def r(i: str, pred: str, f1: float, tok: int, cost: float) -> dict:
        return {"id": i, "pred": pred, "f1": f1, "tokens": tok, "cost": cost}

    arms = {"qwen7b-rag": [r("1", "Paris", 0, 100, 1.0), r("2", "Paris", 1.0, 100, 1.0)],
            "qwen14b-rag": [r("1", "I cannot tell", 0, 100, 2.0), r("2", "Paris", 1.0, 100, 2.0)]}
    rep = report(qs, arms)
    assert probe_score("not-in-corpus", "abstain", arms["qwen7b-rag"][0]) == 0.0
    assert rep["not-in-corpus"]["winner"] == "qwen14b-rag" and not rep["not-in-corpus"]["match"]
    assert rep["single-fact"]["winner"] == "qwen7b-rag" and rep["single-fact"]["match"]  # tie -> cheaper


def test_probe_never_draws_numeric_finance_questions(tmp_path: Path, monkeypatch) -> None:
    from eval import run_probe

    assert not {typ for _, typ in run_probe.DRAW} & run_probe.NUMERIC_TYPES
    monkeypatch.setitem(run_probe.DRAW, ("finance", "metrics-generated"), ("finance-numeric", "qwen7b-rag"))
    out = tmp_path / "p.jsonl"
    out.write_text("")
    with pytest.raises(ValueError, match="D39"):
        run_probe.draw(out=out)
    assert out.read_text() == ""
