"""Live progress of `make phase3`: rows done per result file and arm, mean F1/latency, errors, rough ETA.

    uv run python scripts/phase3_watch.py        # refreshes every 15 s, Ctrl-C to quit
    uv run python scripts/phase3_watch.py --once
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

RUNS = Path("eval/runs")
TARGET = {"general": 300, "finance": 100}  # questions per arm (make phase3: 100/type general, 50/type finance text)


def summarize() -> list[str]:
    lines, remaining_s = [], 0.0
    for p in sorted(RUNS.glob("baseline_*.json")):
        if ".judged" in p.name:
            continue
        domain = "finance" if "finance" in p.name else "general"
        by_mode: dict[str, list[dict]] = defaultdict(list)
        for r in json.loads(p.read_text())["results"]:
            by_mode[r["mode"]].append(r)
        for mode in ("rag", "agent"):
            rows = by_mode.get(mode, [])
            if not rows:
                continue
            n, tot = len(rows), TARGET[domain]
            lat = sum(r["latency"] for r in rows) / n
            remaining_s += max(tot - n, 0) * lat
            lines.append(f"{p.stem:<26} {mode:<6} {n:>3}/{tot:<3} f1={sum(r['f1'] for r in rows) / n:.3f} "
                         f"lat={lat:5.1f}s tok={sum(r['tokens'] for r in rows) // n:>5} "
                         f"err={sum(bool(r['error']) for r in rows)}")
    lines.append(f"started arms only; ETA for those (rows x mean latency): ~{remaining_s / 60:.0f} min "
                 f"(arms not started yet are not counted)")
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        print("\033c" + time.strftime("%H:%M:%S") + "  phase 3 progress\n" + "\n".join(summarize()), flush=True)
        if a.once:
            break
        time.sleep(15)
