"""Warm-up before a demo (parallel-work C4): check /health, then fire one cheap query per arm so
models are loaded and prefix caches are warm. Exit 1 if anything is down or an arm errors.

    python scripts/warmup.py --engine http://localhost:8000
"""
import argparse

import httpx

from engine.api.health import Health, decide_mode
from engine.router.arms import ArmRegistry
from scripts.record_replay import run_and_wait

QUESTION = "Who directed the film Inception?"


def warmup(client: httpx.Client, arms: list[str], question: str = QUESTION) -> dict[str, str]:
    """{arm: 'ok' | 'error: ...'}; the first entry '_health' says whether live mode is possible."""
    health = Health.model_validate(client.get("/health").json())
    out = {"_health": decide_mode(health, replay_available=False)}
    if out["_health"] != "live":
        return out
    for arm in arms:
        try:
            events = run_and_wait(client, {"question": question, "arm": arm}, timeout_s=300)
            stop = events[-1]
            out[arm] = "ok" if stop.type == "stop" and not stop.error else f"error: {stop.error}"
        except (httpx.HTTPError, TimeoutError) as e:
            out[arm] = f"error: {e}"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://localhost:8000")
    a = ap.parse_args()
    with httpx.Client(base_url=a.engine, timeout=30) as c:
        res = warmup(c, [x.name for x in ArmRegistry.from_yaml().arms])
    for k, v in res.items():
        print(f"{k:<16} {v}")
    raise SystemExit(0 if res["_health"] == "live" and all(v == "ok" for k, v in res.items() if k != "_health") else 1)
