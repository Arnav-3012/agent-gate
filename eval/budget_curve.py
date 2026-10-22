"""Budget curve for a cheap -> strong router (D53): quality vs spend, router vs RANDOM at equal spend.

Rules fixed before the judged results (decision.md D53): the escalation grid is fixed (0..100% in
10% steps), the router only RANKS queries (escalate the top r share by held-out P(strong helps)),
so no threshold is tuned on the test fold; every budget is reported, not the best one.

Cost views: `api` = the row's ledger cost (local arms priced as a hosted small model, D4);
`true` = local arms at energy cost (latency x watts x $/kWh, pricing.yaml local_energy), hosted
arms at list price. Escalating pays the strong arm on top of the cheap try (cascade).

    python -m eval.budget_curve <judged files> --domain general --cheap default/rag \\
        --strong claude-haiku-5-5/rag [--view true] [--signal clf|top1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from engine.cost.ledger import Pricing
from eval.benchmark import load_files, score, with_headline_split
from eval.pareto import N_BOOT
from eval.router_clf import features

GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def row_cost(r: dict, arm: str, view: str, energy: dict[str, float]) -> float:
    if view == "true" and not arm.startswith("claude-"):
        return r["latency"] * energy["watts"] / 3.6e6 * energy["usd_per_kwh"]
    return r["cost"]


def curve(rank: np.ndarray, sc: np.ndarray, ss: np.ndarray, cc: np.ndarray, cs: np.ndarray,
          lc: np.ndarray | None = None, ls: np.ndarray | None = None, seed: int = 0) -> list[dict]:
    """rank: higher = escalate first. Per budget share r: router, random (expected), oracle, cost."""
    n, gain = len(sc), ss - sc
    lc = np.zeros(n) if lc is None else lc
    ls = np.zeros(n) if ls is None else ls
    order, best = np.argsort(-rank, kind="stable"), np.argsort(-gain, kind="stable")
    idx = np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))
    out = []
    for r in GRID:
        k = round(r * n)
        esc, ora = np.zeros(n, bool), np.zeros(n, bool)
        esc[order[:k]], ora[best[:k]] = True, True
        router = np.where(esc, ss, sc)
        diff = router - (sc + r * gain)  # vs random escalation of the same share (same expected spend)
        boot = diff[idx].mean(axis=1)
        lat = lc + esc * ls  # cascade: the cheap try always runs first
        out.append({"share": float(r), "usd_per_1k": float((cc + esc * cs).mean() * 1000),
                    "lat_mean": float(lat.mean()), "lat_p95": float(np.percentile(lat, 95)),
                    "router": float(router.mean()), "random": float((sc + r * gain).mean()),
                    "oracle": float(np.where(ora, ss, sc).mean()), "minus_random": float(diff.mean()),
                    "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--view", choices=["api", "true"], default="true")
    ap.add_argument("--signal", choices=["clf", "top1"], default="clf")
    a = ap.parse_args()
    arms = with_headline_split(load_files(a.files))[a.domain]
    ids = sorted(set(arms[a.cheap]) & set(arms[a.strong]))
    cheap, strong = [arms[a.cheap][i] for i in ids], [arms[a.strong][i] for i in ids]
    energy = Pricing.from_yaml().local_energy
    sc, ss = np.array([score(r) for r in cheap]), np.array([score(r) for r in strong])
    cc = np.array([row_cost(r, a.cheap, a.view, energy) for r in cheap])
    cs = np.array([row_cost(r, a.strong, a.view, energy) for r in strong])
    if a.signal == "top1":
        rank = -np.array([r["top1"] for r in cheap])  # low retrieval confidence -> escalate first
    else:  # held-out P(strong beats cheap): each query scored by a model that never saw it
        qpath = Path(f"eval/data/{'finance_questions' if a.domain.startswith('finance') else 'questions'}.jsonl")
        questions = {q["id"]: q["question"] for q in map(json.loads, qpath.read_text().splitlines())}
        y = (ss - sc) > 0
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
        rank = cross_val_predict(clf, features(cheap, questions), y, cv=5, method="predict_proba")[:, 1]
    print(f"{a.domain}: {a.cheap} -> {a.strong}, view={a.view}, signal={a.signal}, n={len(ids)}, "
          f"strong-better rate {np.mean(ss > sc):.0%}")
    print(f"  {'escalate':>8} {'$/1k':>9} {'lat':>5} {'p95':>5} {'router':>7} {'random':>7} {'oracle':>7}  router-random [95% CI]")
    lc, ls = np.array([r["latency"] for r in cheap]), np.array([r["latency"] for r in strong])
    print(f"  fixed {a.strong}: score {ss.mean():.3f}, ${cs.mean() * 1000:.4f}/1k, latency mean {ls.mean():.1f}s p95 {np.percentile(ls, 95):.1f}s")
    for p in curve(rank, sc, ss, cc, cs, lc, ls):
        lo, hi = p["ci"]
        print(f"  {p['share']:>8.0%} {p['usd_per_1k']:>9.4f} {p['lat_mean']:>4.1f}s {p['lat_p95']:>4.1f}s {p['router']:>7.3f} {p['random']:>7.3f} "
              f"{p['oracle']:>7.3f}  {p['minus_random']:+.3f} [{lo:+.3f},{hi:+.3f}] {'REAL' if lo > 0 else ''}")


if __name__ == "__main__":
    main()
