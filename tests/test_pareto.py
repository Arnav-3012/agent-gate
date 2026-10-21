import numpy as np

from eval.pareto import boot_ci, oracle


def test_oracle_picks_cheapest_best_arm() -> None:
    arms = {"rag": {"a": (1.0, 1.0), "b": (0.0, 1.0)}, "agent": {"a": (1.0, 3.0), "b": (0.5, 3.0)}}
    f, c = oracle(arms, ["a", "b"])
    assert list(f) == [1.0, 0.5] and list(c) == [1.0, 3.0]


def test_boot_ci_brackets_mean() -> None:
    m, lo, hi = boot_ci(np.array([0.0, 1.0] * 20), np.random.default_rng(0))
    assert lo < m < hi and m == 0.5


def test_exclude_types_filters_rows(tmp_path) -> None:
    import json

    from eval.pareto import load_arms

    rows = [{"id": "a", "type": "x", "mode": "rag", "f1": 1.0, "cost": 1.0, "latency": 1.0},
            {"id": "b", "type": "metrics-generated", "mode": "rag", "f1": 0.0, "cost": 1.0, "latency": 1.0}]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"model": "m", "results": rows}))
    assert list(load_arms([f], frozenset({"metrics-generated"}))["m/rag"]) == ["a"]
    assert len(load_arms([f])["m/rag"]) == 2
