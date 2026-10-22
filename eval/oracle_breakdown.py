"""Where does the oracle-router gain come from, and how much of it is selection noise?

Per domain: (1) which arm supplies the per-question best score (gain over the best fixed arm,
split by arm); (2) oracle gain with the agent arms removed; (3) a null where arms differ only by a
constant offset: remove each arm's mean, shuffle the residuals across arms WITHIN each question
(question difficulty is kept), add the means back, recompute the oracle gain. That is what picking
the max of noisy arms gives for free; only the excess over it is evidence that the arms really
differ per question.

    python -m eval.oracle_breakdown eval/runs/baseline_*.json [--arms default/rag,claude-sonnet-5-5/rag]
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from eval.benchmark import load_files, score, with_headline_split


def oracle_gain(m: np.ndarray) -> float:
    """m: (arms, questions) scores. Oracle mean minus the best fixed arm's mean."""
    return float(m.max(axis=0).mean() - m.mean(axis=1).max())


def main(paths: list[Path], keep: set[str] | None = None, n_perm: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for dom, arms in with_headline_split(load_files(paths)).items():
        names = sorted(a for a in arms if not a.endswith("/oracle") and (not keep or a in keep))
        if len(names) < 2:
            continue
        ids = sorted(set.intersection(*(set(arms[a]) for a in names)))
        m = np.array([[score(arms[a][i]) for i in ids] for a in names])
        best_fixed = int(m.mean(axis=1).argmax())
        gain_q = m.max(axis=0) - m[best_fixed]
        share = Counter(names[int(m[:, j].argmax())] for j in range(len(ids)) if gain_q[j] > 1e-9)
        no_agent = [k for k, a in enumerate(names) if not a.endswith("/agent")]
        res, mu = m - m.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)
        null = [oracle_gain(mu + rng.permuted(res, axis=0)) for _ in range(n_perm)]
        print(f"\n{dom}: {len(ids)} questions, best fixed = {names[best_fixed]} ({m[best_fixed].mean():.3f})")
        print(f"  oracle gain (all arms)   {oracle_gain(m):+.3f}   noise-only null {np.mean(null):+.3f} "
              f"[{np.percentile(null, 2.5):+.3f},{np.percentile(null, 97.5):+.3f}]  -> excess {oracle_gain(m) - np.mean(null):+.3f}")
        print(f"  oracle gain (no agents)  {oracle_gain(m[no_agent]):+.3f}")
        print(f"  questions where best fixed is beaten: {int((gain_q > 1e-9).sum())}/{len(ids)}; supplied by:")
        for n, c in share.most_common():
            print(f"    {n:<34} {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--arms", default="", help="comma list of arms to keep (default: all)")
    a = ap.parse_args()
    main(a.files, set(filter(None, a.arms.split(","))))
