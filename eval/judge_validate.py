"""Judge validation (parallel-work A3): export a stratified sample for hand labelling, then score
agreement between the judge and the human. Numeric-gold rows are deterministic and excluded.

    python eval/judge_validate.py export eval/runs/smoke_eval_finance.judged.json   # -> judge_sample.csv
    (fill the human_label column with correct / partial / incorrect)
    python eval/judge_validate.py score eval/runs/judge_sample.csv
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("correct", "partial", "incorrect")
FIELDS = ["id", "gold", "pred", "judge_verdict", "human_label"]


def stratified_sample(rows: list[dict], n: int = 30, seed: int = 0) -> list[dict]:
    """Round-robin over judge verdicts so rare verdicts (partial) are not drowned out."""
    by_v: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("verdict") in LABELS:
            by_v[r["verdict"]].append(r)
    rng = random.Random(seed)
    for v in by_v.values():
        rng.shuffle(v)
    out: list[dict] = []
    while len(out) < n and any(by_v.values()):
        for v in sorted(by_v):
            if by_v[v] and len(out) < n:
                out.append(by_v[v].pop())
    return out


def export(judged: Path, out: Path, n: int = 30) -> int:
    rows = stratified_sample(json.loads(judged.read_text())["results"], n)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows({"id": r["id"], "gold": r["gold"], "pred": r["pred"], "judge_verdict": r["verdict"],
                     "human_label": ""} for r in rows)
    return len(rows)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(csv_path: Path) -> dict[str, float]:
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip()]
    j = [r["judge_verdict"] for r in rows]
    h = [r["human_label"].strip().lower() for r in rows]
    return {"n": len(rows), "agreement": sum(x == y for x, y in zip(j, h, strict=True)) / len(rows),
            "kappa": cohen_kappa(j, h)}


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "export":
        out = path.with_name("judge_sample.csv")
        print(f"wrote {export(path, out)} rows -> {out}")
    else:
        print(score(path))
