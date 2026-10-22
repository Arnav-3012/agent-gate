import json

import httpx
import pytest

from engine.llm import anthropic_client
from engine.llm.anthropic_client import AnthropicClient
from engine.llm.client import Message


def _transport(seen: list[dict]) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Paris"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 1_000_000, "output_tokens": 0}})
    return httpx.MockTransport(handler)


def test_chat_shape_and_spend_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anthropic_client, "_spent_usd", 0.0)
    seen: list[dict] = []
    llm = AnthropicClient("claude-sonnet-5-5", api_key="k", max_usd=3.0, transport=_transport(seen))
    msgs = [Message(role="system", content="sys"), Message(role="user", content="q")]
    call = llm.chat(msgs)
    assert call.text == "Paris" and call.prompt_tokens == 1_000_000
    assert "temperature" not in seen[0] and seen[0]["system"] == "sys" and len(seen[0]["messages"]) == 1
    assert anthropic_client.spent_usd() == pytest.approx(2.0)  # 1M input tokens at Sonnet 5.5 $2 (20x anchor)
    llm.chat(msgs)  # 2.0 < 3.0: allowed, now 4.0
    with pytest.raises(SystemExit):
        AnthropicClient("claude-sonnet-5-5", api_key="k", max_usd=3.0, transport=_transport(seen)).chat(msgs)
