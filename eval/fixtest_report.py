"""Fix test T2/T3 report: per file F1 (all / retrieval hit / miss), tokens, stop-reason histogram,
and the same ids from the smoke_eval baseline (before-picture) for comparison.
Usage: python -m eval.fixtest_report eval/runs/fixtest_t2_rag.json [more.json ...] [--base eval/runs/smoke_eval_finance.json]
"""
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


def line(label: str, rs: list[dict]) -> str:
    if not rs:
        return f"{label:<34} n=0"
    hit = [r["f1"] for r in rs if r["gold_found"]]
    miss = [r["f1"] for r in rs if not r["gold_found"]]
    stops = dict(Counter(r["stop"] for r in rs))
    return (f"{label:<34} n={len(rs):<3} F1 {mean(r['f1'] for r in rs):.3f}  "
            f"gold seen {len(hit)}/{len(rs)} (F1 hit {mean(hit) if hit else 0:.3f} / miss "
            f"{mean(miss) if miss else 0:.3f})  tok {mean(r['tokens'] for r in rs):.0f}  "
            f"err {sum(bool(r['error']) for r in rs)}  stop {stops}")


def main() -> None:
    args = sys.argv[1:]
    base = Path(args[args.index("--base") + 1]) if "--base" in args else Path("eval/runs/smoke_eval_finance.json")
    files = [Path(a) for a in args if a.endswith(".json") and Path(a) != base]
    baseline = json.loads(base.read_text())["results"]
    for f in files:
        rs = json.loads(f.read_text())["results"]
        for mode in sorted({r["mode"] for r in rs}):
            new = [r for r in rs if r["mode"] == mode]
            ids = {r["id"] for r in new}
            print(line(f"{f.stem} [{mode}]", new))
            print(line(f"  baseline same ids [{mode}]", [r for r in baseline if r["mode"] == mode and r["id"] in ids]))


if __name__ == "__main__":
    main()
