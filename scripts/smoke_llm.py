"""Smoke test: one chat completion against the local MLX server, prints text, tokens, latency."""
import os
import time

import httpx

BASE = os.getenv("LLM_URL", "http://localhost:8080/v1")
MODEL = os.getenv("LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")

t0 = time.perf_counter()
r = httpx.post(
    f"{BASE}/chat/completions",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Who directed Inception? One short sentence."}],
        "max_tokens": 64,
        "temperature": 0,
    },
    timeout=120,
)
r.raise_for_status()
body = r.json()
dt = time.perf_counter() - t0
print(body["choices"][0]["message"]["content"])
print(f"usage={body['usage']} latency={dt:.2f}s")
