"""OpenAI-compatible chat client (mlx_lm.server) that meters tokens and latency per call."""
import os
import time

import httpx
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class LLMCall(BaseModel):
    """One metered completion. Feeds trace events and the cost ledger."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    latency_s: float
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Meter(BaseModel):
    """Running totals over a query; the agent loop checks this against max_tokens_per_query."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, call: LLMCall) -> None:
        self.calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.cached_tokens += call.cached_tokens
        self.latency_s += call.latency_s


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.meter = Meter()
        self._http = httpx.Client(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
            timeout=timeout,
            transport=transport,
        )

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> LLMCall:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        t0 = time.perf_counter()
        resp = self._http.post("/chat/completions", json=payload)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        usage = body["usage"]
        choice = body["choices"][0]
        call = LLMCall(
            text=choice["message"]["content"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            latency_s=latency,
            finish_reason=choice.get("finish_reason"),
        )
        self.meter.add(call)
        return call

    def ping(self) -> bool:
        """True if the server answers /models; for health checks."""
        try:
            return self._http.get("/models", timeout=3).is_success
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._http.close()
