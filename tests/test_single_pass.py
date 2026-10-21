import json

import httpx

from engine.llm.client import LLMClient
from engine.rag.retriever import Chunk, Hit
from engine.rag.single_pass import run_rag
from engine.tracing.emitter import Emitter


class FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        c = Chunk(chunk_id="c1", title="Inception", text="Directed by Christopher Nolan.", source="squad")
        return [Hit(chunk=c, score=0.9)]


def llm(handler) -> LLMClient:
    return LLMClient(base_url="http://x/v1", transport=httpx.MockTransport(handler))


def ok(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert "Christopher Nolan" in body["messages"][1]["content"]  # context reached the prompt
    return httpx.Response(200, json={
        "choices": [{"message": {"content": " Christopher Nolan\n"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 4},
    })


def test_happy_path_events_and_answer() -> None:
    em = Emitter("t1", [])
    assert run_rag("Who directed Inception?", FakeRetriever(), llm(ok), em) == "Christopher Nolan"
    assert [e.type for e in em.events] == ["retrieve", "answer", "stop"]
    assert [e.seq for e in em.events] == [0, 1, 2]
    stop = em.events[-1]
    assert stop.reason == "rag_done" and stop.total_tokens == 54 and stop.steps == 1
    assert em.events[0].hits[0].chunk_id == "c1"


def test_llm_failure_ends_in_error_stop() -> None:
    em = Emitter("t2", [])
    boom = lambda r: httpx.Response(500)
    assert run_rag("q", FakeRetriever(), llm(boom), em) == ""
    assert [e.type for e in em.events] == ["retrieve", "stop"]
    assert em.events[-1].reason == "error" and "500" in em.events[-1].error
