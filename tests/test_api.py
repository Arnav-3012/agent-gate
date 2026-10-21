import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from engine.agent.loop import AgentConfig
from engine.api.main import Engine, create_app
from engine.cost.ledger import Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import Chunk, Hit


class FakeQdrant:
    def __init__(self, up: bool = True) -> None:
        self.up = up

    def collection_exists(self, name: str) -> bool:
        if not self.up:
            raise ConnectionError
        return True


class FakeRetriever:
    def __init__(self, up: bool = True) -> None:
        self.client = FakeQdrant(up)

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        c = Chunk(chunk_id="c1", title="Inception", text="Directed by Christopher Nolan.", source="squad")
        return [Hit(chunk=c, score=0.9)]


def llm_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": []})
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "Christopher Nolan"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
    })


def make_client(tmp_path: Path, handler=llm_ok, up: bool = True) -> TestClient:
    engine = Engine(
        FakeRetriever(up),
        make_llm=lambda: LLMClient(base_url="http://x/v1", transport=httpx.MockTransport(handler)),
        trace_dir=tmp_path,
        pricing=Pricing(
            reference_model="m", verified_on="2026-10-21",
            usd_per_1m_tokens={"input": 0.10, "output": 0.50},
        ),
        agent_cfg=AgentConfig(),
    )
    return TestClient(create_app(engine))


def test_run_rag_returns_202_and_trace_completes(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        r = c.post("/run", json={"question": "Who directed Inception?"})
        assert r.status_code == 202
        tid = r.json()["trace_id"]
        body = c.get(f"/trace/{tid}").json()  # TestClient runs background tasks before returning
    assert body["done"] and [e["type"] for e in body["events"]] == ["retrieve", "answer", "stop"]
    assert body["events"][1]["llm"]["cost_usd"] > 0 and body["events"][2]["total_cost_usd"] > 0


def test_trace_after_filters_seen_events(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        tid = c.post("/run", json={"question": "q"}).json()["trace_id"]
        events = c.get(f"/trace/{tid}", params={"after": 0}).json()["events"]
    assert [e["seq"] for e in events] == [1, 2]


def test_agent_mode_runs_plan_first(tmp_path: Path) -> None:
    def agent_llm(request: httpx.Request) -> httpx.Response:
        msgs = json.loads(request.content)["messages"]
        text = {"Break": '{"subqueries": ["a"]}', "Decide": '{"confidence": 0.9}'}.get(
            msgs[0]["content"].split()[0], "Christopher Nolan")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        })

    with make_client(tmp_path, agent_llm) as c:
        tid = c.post("/run", json={"question": "q", "mode": "agent"}).json()["trace_id"]
        types = [e["type"] for e in c.get(f"/trace/{tid}").json()["events"]]
    assert types == ["plan", "retrieve", "reflect", "answer", "stop"]


def test_llm_failure_is_an_error_trace_not_a_500(tmp_path: Path) -> None:
    with make_client(tmp_path, lambda r: httpx.Response(500)) as c:
        tid = c.post("/run", json={"question": "q"}).json()["trace_id"]
        body = c.get(f"/trace/{tid}").json()
    assert body["done"] and body["events"][-1]["reason"] == "error"


def test_pending_trace_is_empty_not_404(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        tid = "a" * 32
        (tmp_path / f"{tid}.jsonl").touch()
        assert c.get(f"/trace/{tid}").json() == {"trace_id": tid, "events": [], "done": False}


def test_unknown_and_malicious_ids_404(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        assert c.get("/trace/" + "b" * 32).status_code == 404
        assert c.get("/trace/..%2Fsecret").status_code == 404


def test_run_validates_body(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        assert c.post("/run", json={"question": ""}).status_code == 422
        assert c.post("/run", json={"question": "q", "mode": "router"}).status_code == 422


def test_replay_lists_finished_traces_only(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        done = c.post("/run", json={"question": "q"}).json()["trace_id"]
        (tmp_path / f"{'c' * 32}.jsonl").touch()  # in flight
        (tmp_path / "smoke_rag.jsonl").write_text("not a trace")  # ignored by name
        assert c.get("/replay").json() == [done]


def test_health_reports_each_dependency(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        assert c.get("/health").json() == {"ok": True, "llm": True, "qdrant": True}
    with make_client(tmp_path, lambda r: httpx.Response(503), up=False) as c:
        assert c.get("/health").json() == {"ok": False, "llm": False, "qdrant": False}


def test_two_arms_run_through_their_own_clients_and_are_traced(tmp_path: Path) -> None:
    from engine.router.arms import Arm, ArmRegistry

    arms = ArmRegistry(arms=[
        Arm(name="small-rag", model="small", base_url="http://s/v1", mode="rag"),
        Arm(name="big-rag", model="big", base_url="http://b/v1", mode="rag", price_multiplier=4.0),
    ])
    seen: list[str] = []

    def make(arm: Arm) -> LLMClient:
        seen.append(arm.name)
        return LLMClient(base_url=arm.base_url, model=arm.model, transport=httpx.MockTransport(llm_ok))

    engine = Engine(FakeRetriever(), trace_dir=tmp_path, agent_cfg=AgentConfig(), arms=arms,
                    make_arm_llm=make, pricing=Pricing(reference_model="m", verified_on="2026-10-21",
                                                       usd_per_1m_tokens={"input": 0.10, "output": 0.50}))
    with TestClient(create_app(engine)) as c:
        bodies = [c.get(f"/trace/{c.post('/run', json={'question': 'q', 'arm': a}).json()['trace_id']}").json()
                  for a in ("small-rag", "big-rag")]
        assert c.post("/run", json={"question": "q", "arm": "nope"}).status_code == 422
    assert seen == ["small-rag", "big-rag"]
    assert [b["events"][0]["route"] for b in bodies] == ["small-rag", "big-rag"]
    small, big = (b["events"][-1]["total_cost_usd"] for b in bodies)
    assert big == pytest.approx(4 * small)
