"""Failure drill (parallel-work C3): break one dependency at a time and check the engine's /health
makes the dashboard fall back to replay. Dry-run by default; --execute runs the break/restore commands.

    python scripts/drill.py                 # print the plan
    python scripts/drill.py --execute       # needs the engine, Qdrant and the 7B server running
"""
import argparse
import subprocess
import time

import httpx

from engine.api.health import Health, decide_mode

# name -> (break command, restore command, health we expect while broken)
DRILLS: dict[str, tuple[str, str, Health]] = {
    "llm-down": ("pkill -f 'mlx_lm.server.*--port 8080'", "make llm &", Health(ok=False, llm=False, qdrant=True)),
    "qdrant-down": ("docker compose stop qdrant", "docker compose start qdrant",
                    Health(ok=False, llm=True, qdrant=False)),
}


def fetch_health(engine: str, timeout: float = 5.0) -> Health | None:
    try:
        return Health.model_validate(httpx.get(f"{engine}/health", timeout=timeout).json())
    except (httpx.HTTPError, ValueError):
        return None


def plan() -> list[str]:
    return [f"{n}: break `{b}` -> expect {h.model_dump()} -> mode replay -> restore `{r}`"
            for n, (b, r, h) in DRILLS.items()]


def run_drill(name: str, engine: str, settle_s: float = 3.0) -> bool:
    brk, restore, expect = DRILLS[name]
    subprocess.run(brk, shell=True, check=False)
    try:
        time.sleep(settle_s)
        health = fetch_health(engine)
        ok = health == expect and decide_mode(health, replay_available=True) == "replay"
        print(f"{name}: {'PASS' if ok else 'FAIL'} (health {health})")
        return ok
    finally:
        subprocess.run(restore, shell=True, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--engine", default="http://localhost:8000")
    a = ap.parse_args()
    if not a.execute:
        print("\n".join(plan()), "\n(dry run; pass --execute)")
    else:
        results = [run_drill(n, a.engine) for n in DRILLS]  # every drill runs (and restores), even after a FAIL
        raise SystemExit(0 if all(results) else 1)
