"""Cost-quality Pareto table with bootstrap CIs from one or more eval result files.

Arms are "<model>/<mode>" joined on question id. Also prints the oracle router (cheapest arm that
reaches the question's best F1): its F1/cost is the ceiling any learned router is measured against.
"""
import json
import sys
from pathlib import Path

import numpy as np

N_BOOT = 2000


def load_arms(paths: list[Path]) -> dict[str, dict[str, tuple[float, ...]]]:
    """{arm: {question_id: (f1, cost, latency_s)}}; arm name = model tail + mode."""
    arms: dict[str, dict[str, tuple[float, ...]]] = {}
    for p in paths:
        d = json.loads(p.read_text())
        model = (d.get("model") or "default").split("/")[-1]
        for r in d["results"]:
            arms.setdefault(f"{model}/{r['mode']}", {})[r["id"]] = (r.get("score", r["f1"]), r["cost"], r["latency"])
    return arms


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), *np.percentile(means, [2.5, 97.5])


def oracle(arms: dict[str, dict[str, tuple[float, ...]]], ids: list[str], eps: float = 1e-9):
    """Per question: cheapest arm whose F1 is within eps of the best."""
    picks = []
    for i in ids:
        best = max(a[i][0] for a in arms.values())
        picks.append(min((a[i] for a in arms.values() if a[i][0] >= best - eps), key=lambda t: t[1]))
    return np.array([p[0] for p in picks]), np.array([p[1] for p in picks])


def main(paths: list[Path]) -> None:
    arms = load_arms(paths)
    ids = sorted(set.intersection(*(set(a) for a in arms.values())))
    rng = np.random.default_rng(0)
    print(f"{len(ids)} questions, {N_BOOT} bootstrap resamples\n")
    print(f"{'arm':<32} {'F1':>6} {'95% CI':>15} {'$/1k q':>9} {'lat s':>6}   (cost modelled, D27/D29)")
    rows = {n: (*(np.array([a[i][k] for i in ids]) for k in range(3)),) for n, a in arms.items()}
    f_o, c_o = oracle(arms, ids)
    rows["oracle-router"] = (f_o, c_o, None)  # latency of the picked arm is not tracked
    for name, (f, c, lat) in rows.items():
        m, lo, hi = boot_ci(f, rng)
        print(f"{name:<32} {m:>6.3f} [{lo:.3f},{hi:.3f}] {c.mean() * 1000:>9.4f} "
              f"{'' if lat is None else format(lat.mean(), '6.1f'):>6}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:] if Path(p).exists()])
