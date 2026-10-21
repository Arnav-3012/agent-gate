import httpx

from engine.llm.client import LLMClient, Message


def _client(captured: list[dict]) -> LLMClient:
    def handler(req: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        )

    return LLMClient(base_url="http://test/v1", model="m", transport=httpx.MockTransport(handler))


def test_chat_meters_tokens_and_latency() -> None:
    sent: list[dict] = []
    llm = _client(sent)
    call = llm.chat([Message(role="user", content="q")], max_tokens=8, stop=["\n"])
    assert call.text == "hi"
    assert (call.prompt_tokens, call.completion_tokens, call.cached_tokens) == (10, 3, 4)
    assert call.total_tokens == 13 and call.latency_s >= 0
    assert sent[0]["max_tokens"] == 8 and sent[0]["stop"] == ["\n"]


def test_meter_accumulates() -> None:
    llm = _client([])
    for _ in range(2):
        llm.chat([Message(role="user", content="q")])
    m = llm.meter
    assert (m.calls, m.prompt_tokens, m.completion_tokens, m.cached_tokens) == (2, 20, 6, 8)
    assert m.total_tokens == 26


def test_http_error_raises_and_is_not_metered() -> None:
    llm = LLMClient(
        base_url="http://test/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    try:
        llm.chat([Message(role="user", content="q")])
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected HTTPStatusError")
    assert llm.meter.calls == 0
