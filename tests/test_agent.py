import json

import httpx
import pytest

from engine.agent.loop import AgentConfig, run_agent
from engine.llm.client import LLMClient
from engine.rag.retriever import Chunk, Hit
from engine.tracing.emitter import Emitter

PLAN = '{"subqueries": ["who directed Inception", "birthplace of Nolan"]}'


def chunk_hit(cid: str) -> Hit:
    return Hit(chunk=Chunk(chunk_id=cid, title=f"T{cid}", text=f"text {cid}", source="hotpotqa"),
               score=0.9)


class FakeRetriever:
    """Returns the chunk ids registered for a query, empty list otherwise."""

    def __init__(self, table: dict[str, list[str]]) -> None:
        self.table, self.queries = table, []

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        self.queries.append(query)
        return [chunk_hit(c) for c in self.table.get(query, [])]


def scripted(replies: list[str], tokens: int = 10) -> LLMClient:
    """LLM that returns replies in order; fails the test if the loop makes extra calls."""
    it = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": next(it)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tokens, "completion_tokens": 1},
        })

    return LLMClient(base_url="http://x/v1", transport=httpx.MockTransport(handler))


def refl(conf: float, nxt: str = "") -> str:
    return json.dumps({"confidence": conf, "next_query": nxt})


def run(replies, table, cfg=None, tokens=10):
    em = Emitter("t", [])
    r = FakeRetriever(table)
    ans = run_agent("Q?", r, scripted(replies, tokens), em, cfg or AgentConfig())
    return ans, em, r


def types(em: Emitter) -> list[str]:
    return [e.type for e in em.events]


def test_confidence_stop_two_hops() -> None:
    table = {"who directed Inception": ["a", "b"], "Nolan born": ["c"]}
    ans, em, r = run([PLAN, refl(0.3, "Nolan born"), refl(0.9), "London"], table)
    assert ans == "London"
    assert types(em) == ["plan", "retrieve", "reflect", "retrieve", "reflect", "answer", "stop"]
    assert r.queries == ["who directed Inception", "Nolan born"]  # reflector's query jumps the queue
    assert em.events[-1].reason == "confidence" and em.events[-1].steps == 2
    assert [e.seq for e in em.events] == list(range(7))
    assert [e.next_query for e in em.events if e.type == "reflect"] == ["Nolan born", ""]


def test_no_new_evidence_stop() -> None:
    table = {"who directed Inception": ["a"], "birthplace of Nolan": ["a"]}  # 2nd step repeats
    _, em, _ = run([PLAN, refl(0.2), refl(0.2), "x"], table)
    reflects = [e for e in em.events if e.type == "reflect"]
    assert [e.new_evidence for e in reflects] == [1, 0]
    assert reflects[-1].verdict == "stop" and em.events[-1].reason == "no_new_evidence"


def test_exhausted_queue_stops() -> None:
    _, em, _ = run(['{"subqueries": ["only"]}', refl(0.2), "x"], {"only": ["a"]})
    assert em.events[-1].reason == "no_new_evidence" and em.events[-1].steps == 1


def test_max_steps_cap() -> None:
    cfg = AgentConfig(max_steps=2)
    table = {"who directed Inception": ["a"], "birthplace of Nolan": ["b"]}
    _, em, _ = run([PLAN, refl(0.2), refl(0.2, "more"), "x"], table, cfg)
    stop = em.events[-1]
    assert stop.reason == "max_steps" and stop.steps == 2
    assert [e.verdict for e in em.events if e.type == "reflect"] == ["continue", "continue"]


def test_projected_cap_stops_before_overshoot() -> None:
    # every call costs 101 tokens; after step 1 (~202 spent) another step + answer would exceed 450
    cfg = AgentConfig(max_tokens=450)
    table = {"who directed Inception": ["a"], "birthplace of Nolan": ["b"]}
    _, em, _ = run([PLAN, refl(0.2), "x"], table, cfg, tokens=100)
    assert em.events[-1].reason == "max_tokens" and em.events[-1].steps == 1
    assert em.events[-1].total_tokens <= 450


def test_max_tokens_cap_skips_further_steps_but_still_answers() -> None:
    cfg = AgentConfig(max_tokens=50)
    _, em, _ = run([PLAN, "x"], {}, cfg, tokens=60)  # plan alone blows the budget
    assert types(em) == ["plan", "answer", "stop"]
    assert em.events[-1].reason == "max_tokens" and em.events[-1].steps == 0


def test_json_retry_then_success_aggregates_stats() -> None:
    table = {"who directed Inception": ["a"]}
    _, em, _ = run(["not json", PLAN, refl(0.95), "x"], table)
    assert em.events[0].llm.prompt_tokens == 20  # both attempts counted
    assert em.events[-1].reason == "confidence"


def test_bad_json_twice_ends_in_error_stop() -> None:
    ans, em, _ = run(["nope", "still nope"], {})
    assert ans == "" and types(em) == ["stop"]
    assert em.events[0].reason == "error" and "_Plan" in em.events[0].error


def test_out_of_range_confidence_is_rejected() -> None:
    table = {"who directed Inception": ["a"]}
    _, em, _ = run([PLAN, refl(85), refl(0.9), "x"], table)  # first reflect retried
    assert em.events[-1].reason == "confidence"


def test_cfg_from_yaml_matches_file() -> None:
    cfg = AgentConfig.from_yaml()
    assert (cfg.max_steps, cfg.max_tokens, cfg.confidence_threshold) == (4, 6000, 0.8)


@pytest.mark.parametrize("bad", ['{"subqueries": []}', '{"subqueries": ["a","b","c","d"]}'])
def test_plan_bounds(bad: str) -> None:
    _, em, _ = run([bad, bad], {})
    assert em.events[0].reason == "error"
