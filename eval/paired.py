"""Paired bootstrap of arm differences: resample QUESTIONS, recompute mean(score_A - score_B).

Per-arm CIs that overlap say nothing about whether A beats B (both move with question difficulty);
the CI of the paired difference does. "real" = the 95% CI of the difference excludes 0.
Reads the same files as benchmark.py (judged files supersede, finance headline split).

    python -m eval.paired eval/runs/baseline_*.json [--domain finance]
"""
import itertools
import sys
from pathlib import Path

import numpy as np

from eval.benchmark import load_files, score, with_headline_split
from eval.pareto import N_BOOT


def paired_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """(mean diff, lo, hi) of a - b over the same questions."""
    d = a - b
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def compare(arms: dict[str, dict[str, dict]], seed: int = 0) -> list[dict]:
    """All unordered arm pairs on the questions both answered; oracle mode is a diagnostic, skipped."""
    rng = np.random.default_rng(seed)
    names = sorted(a for a in arms if not a.endswith("/oracle"))
    out = []
    for x, y in itertools.combinations(names, 2):
        ids = sorted(set(arms[x]) & set(arms[y]))
        if len(ids) < 2:
            continue
        d, lo, hi = paired_diff(np.array([score(arms[x][i]) for i in ids]),
                                np.array([score(arms[y][i]) for i in ids]), rng)
        out.append({"a": x, "b": y, "n": len(ids), "diff": d, "lo": lo, "hi": hi, "real": lo > 0 or hi < 0})
    return out


def main(paths: list[Path], only: str | None = None) -> None:
    for domain, arms in with_headline_split(load_files(paths)).items():
        if only and domain != only:
            continue
        print(f"\n{domain}: paired bootstrap of score(A) - score(B), {N_BOOT} resamples")
        for r in compare(arms):
            print(f"  {r['a']:<22} - {r['b']:<22} n={r['n']:<4} {r['diff']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}] {'REAL' if r['real'] else 'within noise'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dom = args.pop(args.index("--domain") + 1) if "--domain" in args else None
    if "--domain" in args:
        args.remove("--domain")
    main([Path(a) for a in args], dom)
