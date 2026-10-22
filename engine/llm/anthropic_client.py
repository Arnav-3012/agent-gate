"""Anthropic Messages API answerer with the LLMClient interface (D52 cross-tier arms).

Spend is capped PER PROCESS (all clients share one counter) at AGENTGATE_MAX_USD, priced with the
same ledger as every arm, so a bug or a resumed run cannot run up the bill. No `temperature` is
sent (the 5.5 models reject it) and thinking is switched off: it is on by default and its tokens
count against max_tokens, which gave empty answers (mistake.md).
"""
import os
import time

import httpx

from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMCall, Message, Meter
from engine.tracing.schema import LLMStats

URL = "https://api.anthropic.com/v1/messages"
_spent_usd = 0.0


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None, max_usd: float | None = None,
                 transport: httpx.BaseTransport | None = None, timeout: float = 120.0) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("hosted arm needs ANTHROPIC_API_KEY (set -a; source .env; set +a)")
        self.model, self.meter = model, Meter()
        self.max_usd = max_usd if max_usd is not None else float(os.getenv("AGENTGATE_MAX_USD", "1.0"))
        self._ledger = Ledger(Pricing.from_yaml(), model)
        self._http = httpx.Client(timeout=timeout, transport=transport, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})

    def chat(self, messages: list[Message], max_tokens: int = 256, temperature: float = 0.0,
             stop: list[str] | None = None) -> LLMCall:
        global _spent_usd
        if _spent_usd >= self.max_usd:
            raise SystemExit(f"hosted spend cap hit: ~${_spent_usd:.4f} >= ${self.max_usd}")
        body: dict = {"model": self.model, "max_tokens": max_tokens,
                      "system": "\n".join(m.content for m in messages if m.role == "system"),
                      "messages": [m.model_dump() for m in messages if m.role != "system"]}
        body["thinking"] = {"type": "between_tools" if "sonnet" in self.model else "disabled"}  # else thinking eats max_tokens -> empty answer
        if stop:
            body["stop_sequences"] = stop
        t0 = time.perf_counter()
        resp = self._http.post(URL, json=body)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        out = resp.json()
        call = LLMCall(text="".join(b["text"] for b in out["content"] if b["type"] == "text"),
                       prompt_tokens=out["usage"]["input_tokens"],
                       completion_tokens=out["usage"]["output_tokens"],
                       latency_s=latency, finish_reason=out.get("stop_reason"))
        _spent_usd += self._ledger.price(LLMStats(prompt_tokens=call.prompt_tokens,
                                                  completion_tokens=call.completion_tokens, latency_s=latency))
        self.meter.add(call)
        return call


def spent_usd() -> float:
    return _spent_usd
