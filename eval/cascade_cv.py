"""Held-out scoring of a 2-rung cascade: answer with the cheap arm, escalate to the strong arm when
a serve-time signal (default: first-retrieval top-1 score) is below a threshold.

Thresholds are fitted on training folds only and scored on the held-out fold, so every question is
predicted by a threshold that never saw it (picking the threshold on the test set is the classic
trap). Utility on a fold = mean score - lam * mean cost, with lam the break-even slope
(strong's extra score per extra dollar) so "always cheap" and "always strong" tie at zero and the
cascade wins only if the signal separates questions that need the strong arm. The control is
RANDOM escalation at the same rate: a useful signal must beat it, not just beat the cheap arm.

    python -m eval.cascade_cv eval/runs/baseline_*.json --domain finance --cheap qwen2.5-7b-instruct-4bit/rag \\
        --strong qwen2.5-14b-instruct-4bit/rag [--signal top1]
"""
import argparse
from pathlib import Path

import numpy as np

from eval.benchmark import load_files, score, with_headline_split
from eval.pareto import N_BOOT


def auc(signal: np.ndarray, positive: np.ndarray) -> float:
    """P(signal of a random positive < signal of a random negative): low signal should flag the
    questions where the strong arm helps. 0.5 = no information. Ties count half."""
    pos, neg = signal[positive], signal[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] < neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def fit_threshold(sig: np.ndarray, gain: np.ndarray, extra_cost: np.ndarray) -> float:
    """Threshold t (escalate when sig < t) maximising mean(gain - lam*extra_cost) on the fit set,
    with lam = break-even slope. Candidates are the observed signal values."""
    lam = gain.mean() / extra_cost.mean() if extra_cost.mean() > 0 and gain.mean() > 0 else 0.0
    net = gain - lam * extra_cost  # per-question benefit of escalating
    cands = np.unique(sig)
    util = [net[sig < t].sum() for t in cands]
    return float(cands[int(np.argmax(util))]) if util and max(util) > 0 else float(sig.min())  # never escalate


def cross_validate(cheap: list[dict], strong: list[dict], signal: str = "top1", k: int = 5,
                   seed: int = 0) -> dict:
    """cheap/strong are aligned row lists (same question order). Returns held-out metrics."""
    n = len(cheap)
    sig = np.array([r[signal] for r in cheap])
    sc, ss = np.array([score(r) for r in cheap]), np.array([score(r) for r in strong])
    cc, cs = np.array([r["cost"] for r in cheap]), np.array([r["cost"] for r in strong])
    gain, extra = ss - sc, cs  # escalating pays the strong arm in full on top of the cheap try
    folds = np.random.default_rng(seed).permutation(n) % k
    esc = np.zeros(n, dtype=bool)
    for f in range(k):
        t = fit_threshold(sig[folds != f], gain[folds != f], extra[folds != f])
        esc[folds == f] = sig[folds == f] < t
    return summarize(sig, sc, ss, cc, cs, esc, seed)


def summarize(sig: np.ndarray, sc: np.ndarray, ss: np.ndarray, cc: np.ndarray, cs: np.ndarray,
              esc: np.ndarray, seed: int = 0) -> dict:
    """Held-out metrics of a given escalation decision vs cheap-only and RANDOM escalation at the same rate."""
    n, gain = len(sc), ss - sc
    casc_score = np.where(esc, ss, sc)
    casc_cost = cc + np.where(esc, cs, 0.0)
    rate = float(esc.mean())
    chance_score = sc + rate * gain  # expected score of escalating a random `rate` share
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))

    def ci(x: np.ndarray) -> tuple[float, float, float]:
        m = x[idx].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    return {
        "n": n, "escalation_rate": rate, "auc_low_signal_flags_strong_helps": auc(sig, gain > 0),
        "score": {"cheap": float(sc.mean()), "strong": float(ss.mean()), "cascade": float(casc_score.mean())},
        "usd_per_1k": {"cheap": float(cc.mean() * 1000), "strong": float(cs.mean() * 1000),
                       "cascade": float(casc_cost.mean() * 1000)},
        "cascade_minus_cheap": ci(casc_score - sc),
        "cascade_minus_random_same_rate": ci(casc_score - chance_score),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--signal", default="top1", choices=["top1", "gap12", "entropy", "sc_agree"])
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    arms = with_headline_split(load_files(a.files))[a.domain]
    ids = sorted(set(arms[a.cheap]) & set(arms[a.strong]))
    if any(a.signal not in arms[a.cheap][i] for i in ids):
        raise SystemExit(f"rows lack '{a.signal}': it is recorded by run_baselines.py runs only")
    r = cross_validate([arms[a.cheap][i] for i in ids], [arms[a.strong][i] for i in ids], a.signal, a.folds)
    print(f"{a.domain}: {a.cheap} -> {a.strong}, signal={a.signal}, n={r['n']}, {a.folds}-fold held-out")
    print(f"  score  cheap {r['score']['cheap']:.3f}  strong {r['score']['strong']:.3f}  cascade {r['score']['cascade']:.3f}")
    print(f"  $/1k   cheap {r['usd_per_1k']['cheap']:.4f}  strong {r['usd_per_1k']['strong']:.4f}  cascade {r['usd_per_1k']['cascade']:.4f}")
    print(f"  escalation rate {r['escalation_rate']:.0%}  (health target ~5-30%)  AUC {r['auc_low_signal_flags_strong_helps']:.2f}")
    for name in ("cascade_minus_cheap", "cascade_minus_random_same_rate"):
        m, lo, hi = r[name]
        print(f"  {name:<32} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] {'REAL' if lo > 0 else 'within noise'}")


if __name__ == "__main__":
    main()
