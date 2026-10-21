import json

import httpx

from engine.llm.client import LLMClient
from eval.judge import HostedJudge, judge_row, parse_verdict
from eval.metrics import is_numeric_gold


def test_parse_verdict_checks_incorrect_before_correct() -> None:
    assert parse_verdict("Incorrect.") == "incorrect"
    assert parse_verdict("partial") == "partial"
    assert parse_verdict("The answer is correct") == "correct"
    assert parse_verdict("no idea") is None


def test_numeric_vs_sentence_gold() -> None:
    assert is_numeric_gold("$1577.00") and is_numeric_gold("10.3%")
    assert not is_numeric_gold("Yes. Verizon's capital intensity ratio was approximately 17%")


def test_judge_row_maps_reply_and_defaults_to_incorrect() -> None:
    def make(reply: str) -> LLMClient:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "Reference answer: gold" in json.loads(request.content)["messages"][1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": reply},
                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return LLMClient(base_url="http://x/v1", transport=httpx.MockTransport(handler))

    assert judge_row(make("correct"), "q", "gold", "pred") == "correct"
    assert judge_row(make("hmm"), "q", "gold", "pred") == "incorrect"


def test_hosted_judge_calls_messages_api_and_costs_tokens() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(headers=request.headers, body=json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "partial"}],
                                         "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}})

    judge = HostedJudge("claude-haiku-5-5", api_key="k", transport=httpx.MockTransport(handler))
    assert judge_row(judge, "q", "gold", "pred") == "partial"
    assert seen["headers"]["x-api-key"] == "k" and seen["body"]["system"].startswith("You grade")
    assert all(m["role"] != "system" for m in seen["body"]["messages"])
    assert abs(judge.cost_usd() - 0.60) < 1e-9  # 1M in at $0.10 + 1M out at $0.50


def test_hosted_judge_refuses_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import pytest
    with pytest.raises(SystemExit):
        HostedJudge()


def test_hosted_judge_aborts_past_spend_cap() -> None:
    import pytest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": "correct"}],
                                         "usage": {"input_tokens": 10_000_000, "output_tokens": 0}})

    judge = HostedJudge("claude-haiku-5-5", api_key="k", transport=httpx.MockTransport(handler), max_usd=0.5)
    judge_row(judge, "q", "g", "p")  # first call costs $1.00 by the mock's token count
    with pytest.raises(SystemExit):
        judge_row(judge, "q", "g", "p")
