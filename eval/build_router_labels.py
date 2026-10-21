"""Router training labels from arm result files: per question, the cheapest arm whose score is
within eps of the best (D-plan: RAG suffices if F1_rag >= F1_best - eps). Arms missing a question
are ignored for it; questions answered by fewer than --min-arms arms are dropped."""
import argparse
import json
from pathlib import Path

import yaml

from eval.benchmark import Row, load_files, score


def build_labels(arms: dict[str, dict[str, Row]], eps: float = 0.05, min_arms: int = 2) -> dict[str, str]:
    routable = {n: a for n, a in arms.items() if not n.endswith("/oracle")}
    labels: dict[str, str] = {}
    for qid in sorted(set().union(*(set(a) for a in routable.values())) if routable else []):
        rows = {n: a[qid] for n, a in routable.items() if qid in a}
        if len(rows) < min_arms:
            continue
        best = max(score(r) for r in rows.values())
        ok = [(r["cost"], n) for n, r in rows.items() if score(r) >= best - eps]
        labels[qid] = min(ok)[1]  # ties on cost break by arm name
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--eps", type=float, default=yaml.safe_load(Path("config/router.yaml").read_text())["epsilon_f1"])
    ap.add_argument("--out", type=Path, default=Path("eval/router_labels.json"))
    a = ap.parse_args()
    arms = load_files(a.files or sorted(Path("eval/runs").glob("*.json")))[a.domain]
    labels = build_labels(arms, a.eps)
    a.out.write_text(json.dumps(labels, indent=1))
    counts = {n: list(labels.values()).count(n) for n in sorted(set(labels.values()))}
    print(f"{len(labels)} labels -> {a.out}: {counts}")


if __name__ == "__main__":
    main()
