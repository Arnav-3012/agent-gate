"""Aggregates eval/runs/*.json into eval/results.json (the file the dashboard Benchmark page reads).

Per domain (finance if the file name says so, else general), per arm "<model>/<mode>" on the
questions every routable arm answered: F1 + bootstrap CI, EM, $/1k queries, p50/p95 latency, F1 by
type and by level. Also the oracle router (cheapest arm reaching each question's best F1) and its
gain over the best fixed arm. `oracle` mode is a diagnostic arm: reported, never routed to.
A <name>.judged.json replaces <name>.json (judge scores supersede token F1).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.pareto import boot_ci

Row = dict


def load_files(paths: list[Path]) -> dict[str, dict[str, dict[str, Row]]]:
    """{domain: {arm: {question_id: row}}}; later files override earlier ones for the same arm+id."""
    judged = {p.name.removesuffix(".judged.json") for p in paths if p.name.endswith(".judged.json")}
    out: dict[str, dict[str, dict[str, Row]]] = defaultdict(lambda: defaultdict(dict))
    for p in sorted(paths):
        if p.name.removesuffix(".json") in judged:
            continue
        d = json.loads(p.read_text())
        domain = "finance" if "finance" in p.name else "general"
        model = (d.get("model") or "default").split("/")[-1]
        for r in d["results"]:
            out[domain][f"{model}/{r['mode']}"][r["id"]] = r
    return out


def score(r: Row) -> float:
    return r.get("score", r["f1"])


def group_f1(rows: list[Row], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "unknown"))].append(score(r))
    return {k: {"f1": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())}


def arm_stats(rows: list[Row], rng: np.random.Generator) -> dict:
    f = np.array([score(r) for r in rows])
    lat = np.array([r["latency"] for r in rows])
    m, lo, hi = boot_ci(f, rng)
    return {"f1": m, "f1_ci": [lo, hi], "em": float(np.mean([r["em"] for r in rows])),
            "cost_per_1k": float(np.mean([r["cost"] for r in rows]) * 1000),
            "latency_p50": float(np.percentile(lat, 50)), "latency_p95": float(np.percentile(lat, 95)),
            "by_type": group_f1(rows, "type"), "by_level": group_f1(rows, "level"), "n": len(rows)}


def oracle_router(arms: dict[str, dict[str, Row]], ids: list[str], eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Per question the cheapest arm within eps of the best score; returns (scores, costs)."""
    f, c = [], []
    for i in ids:
        best = max(score(a[i]) for a in arms.values())
        pick = min((a[i] for a in arms.values() if score(a[i]) >= best - eps), key=lambda r: r["cost"])
        f.append(score(pick))
        c.append(pick["cost"])
    return np.array(f), np.array(c)


def benchmark(files: dict[str, dict[str, dict[str, Row]]], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    result: dict = {}
    for domain, arms in sorted(files.items()):
        routable = {n: a for n, a in arms.items() if not n.endswith("/oracle")}
        ids = sorted(set.intersection(*(set(a) for a in routable.values()))) if routable else []
        stats = {n: arm_stats([a[i] for i in ids], rng) for n, a in routable.items() if ids}
        for n, a in arms.items():
            if n.endswith("/oracle"):
                common = [i for i in ids if i in a]
                if common:
                    stats[n] = arm_stats([a[i] for i in common], rng)
        entry: dict = {"n": len(ids), "arms": stats}
        if ids:
            f, c = oracle_router(routable, ids)
            m, lo, hi = boot_ci(f, rng)
            best_fixed = max(s["f1"] for n, s in stats.items() if n in routable)
            entry["oracle_router"] = {"f1": m, "f1_ci": [lo, hi], "cost_per_1k": float(c.mean() * 1000),
                                      "gain_over_best_fixed": m - best_fixed}
        result[domain] = entry
    return result


def main(paths: list[Path]) -> None:
    res = benchmark(load_files(paths))
    out = Path("eval/results.json")
    out.write_text(json.dumps(res, indent=1))
    for domain, e in res.items():
        print(f"\n{domain}: {e['n']} questions")
        for n, s in e["arms"].items():
            print(f"  {n:<32} F1 {s['f1']:.3f} [{s['f1_ci'][0]:.3f},{s['f1_ci'][1]:.3f}] "
                  f"${s['cost_per_1k']:.4f}/1k  p50 {s['latency_p50']:.1f}s p95 {s['latency_p95']:.1f}s")
        if "oracle_router" in e:
            o = e["oracle_router"]
            print(f"  {'oracle-router':<32} F1 {o['f1']:.3f} ${o['cost_per_1k']:.4f}/1k  "
                  f"gain over best fixed {o['gain_over_best_fixed']:+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]] or sorted(Path("eval/runs").glob("*.json")))
