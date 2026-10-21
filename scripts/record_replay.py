"""Record demo questions as replay traces (parallel-work C1): POST /run, poll /trace/{id} until the
stop event, save the events as traces/replay/<n>-<slug>.jsonl (same format the engine writes and the
dashboard replay mode reads). Run against a live engine: python scripts/record_replay.py q.txt --arm qwen7b-rag"""
import argparse
import re
import time
from pathlib import Path

import httpx

from engine.tracing.schema import TraceEvent, event_adapter


def run_and_wait(client: httpx.Client, body: dict, timeout_s: float = 300, poll_s: float = 0.5) -> list[TraceEvent]:
    """Starts a run and returns all its events once done; TimeoutError if no stop arrives."""
    r = client.post("/run", json=body)
    r.raise_for_status()
    trace_id = r.json()["trace_id"]
    deadline = time.monotonic() + timeout_s
    while True:
        view = client.get(f"/trace/{trace_id}").json()
        if view["done"]:
            return [event_adapter.validate_python(e) for e in view["events"]]
        if time.monotonic() > deadline:
            raise TimeoutError(f"trace {trace_id} not done after {timeout_s}s")
        time.sleep(poll_s)


def record(client: httpx.Client, questions: list[str], out_dir: Path, arm: str | None = None,
           poll_s: float = 0.5) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, q in enumerate(questions, 1):
        body = {"question": q, **({"arm": arm} if arm else {})}
        events = run_and_wait(client, body, poll_s=poll_s)
        p = out_dir / f"{i:02d}-{re.sub(r'[^a-z0-9]+', '-', q.lower()).strip('-')[:40]}.jsonl"
        p.write_text("".join(event_adapter.dump_json(e).decode() + "\n" for e in events))
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path, help="text file, one question per line")
    ap.add_argument("--engine", default="http://localhost:8000")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", type=Path, default=Path("traces/replay"))
    a = ap.parse_args()
    qs = [l.strip() for l in a.questions.read_text().splitlines() if l.strip()]
    with httpx.Client(base_url=a.engine, timeout=30) as c:
        for p in record(c, qs, a.out, a.arm):
            print("saved", p)
