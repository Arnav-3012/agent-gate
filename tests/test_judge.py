import json

import httpx

from engine.llm.client import LLMClient
from eval.judge import judge_row, parse_verdict
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
