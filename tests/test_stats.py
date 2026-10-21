import numpy as np

from eval.cascade_cv import auc, cross_validate, fit_threshold
from eval.paired import compare, paired_diff


def rows(scores: list[float], cost: float, sig: list[float] | None = None) -> list[dict]:
    return [{"f1": s, "cost": cost, "top1": (sig or [0.0] * len(scores))[i]} for i, s in enumerate(scores)]


def test_paired_diff_detects_consistent_gap_that_unpaired_would_miss() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, 100)  # wide question-difficulty spread -> per-arm CIs overlap
    m, lo, hi = paired_diff(base + 0.05, base, rng)
    assert abs(m - 0.05) < 1e-9 and lo > 0.04 and hi < 0.06


def test_compare_skips_oracle_and_flags_noise() -> None:
    arms = {f"m/{mode}": {f"q{i}": {"f1": v} for i, v in enumerate(vals)}
            for mode, vals in {"rag": [1, 0, 1, 0] * 10, "agent": [0, 1, 0, 1] * 10, "oracle": [1] * 40}.items()}
    out = compare(arms)
    assert len(out) == 1 and out[0]["n"] == 40 and not out[0]["real"]


def test_auc_perfect_none_and_ties() -> None:
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(s, np.array([True, True, False, False])) == 1.0
    assert auc(s, np.array([False, False, True, True])) == 0.0
    assert auc(np.ones(4), np.array([True, False, True, False])) == 0.5


def test_fit_threshold_escalates_only_where_it_pays() -> None:
    sig = np.array([0.1, 0.2, 0.8, 0.9])
    gain = np.array([1.0, 1.0, 0.0, 0.0])
    t = fit_threshold(sig, gain, np.full(4, 1.0))
    assert 0.2 < t <= 0.8


def test_fit_threshold_never_escalates_when_strong_is_no_better() -> None:
    sig = np.array([0.1, 0.5, 0.9])
    assert fit_threshold(sig, np.zeros(3), np.ones(3)) == 0.1  # nothing is below the minimum


def test_informative_signal_beats_random_escalation() -> None:
    n = 100
    sig = [i / n for i in range(n)]  # low signal = cheap arm fails
    cheap = rows([0.0 if i < 30 else 1.0 for i in range(n)], 1e-4, sig)
    strong = rows([1.0] * n, 2e-4, sig)
    r = cross_validate(cheap, strong, "top1")
    assert r["auc_low_signal_flags_strong_helps"] > 0.95
    assert r["score"]["cascade"] > 0.9 and r["usd_per_1k"]["cascade"] < r["usd_per_1k"]["strong"]
    assert r["cascade_minus_random_same_rate"][1] > 0


def test_useless_signal_is_not_credited() -> None:
    rng = np.random.default_rng(1)
    n = 200
    sc = (rng.uniform(size=n) > 0.5).astype(float)
    r = cross_validate(rows(list(sc), 1e-4, list(rng.uniform(size=n))), rows([1.0] * n, 2e-4), "top1")
    assert r["cascade_minus_random_same_rate"][1] <= 0.05  # no real edge over chance
