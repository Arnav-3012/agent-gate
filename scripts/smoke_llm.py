"""Smoke test: two metered chat calls via LLMClient (second shows warm latency + prompt cache)."""
from engine.llm.client import LLMClient, Message

llm = LLMClient()
msgs = [Message(role="user", content="Who directed Inception? One short sentence.")]
for label in ("first", "second"):
    c = llm.chat(msgs, max_tokens=64)
    print(f"[{label}] {c.text!r} prompt={c.prompt_tokens} completion={c.completion_tokens} "
          f"cached={c.cached_tokens} latency={c.latency_s:.2f}s finish={c.finish_reason}")
print("meter:", llm.meter.model_dump())
