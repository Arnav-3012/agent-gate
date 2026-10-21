import csv
import json
from pathlib import Path

import numpy as np
import pytest

from engine.router.bandit import LinUCB, reward
from eval.judge_validate import cohen_kappa, export, score, stratified_sample
from eval.pareto import latency_stats
from eval.replay_bandit import replay
from router_model.train import load, route, save, train


def test_latency_stats() -> None:
    s = latency_stats(np.arange(1.0, 101.0), slo_s=90)
    assert s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05) and s["within_slo"] == 0.9
    assert "within_slo" not in latency_stats(np.array([1.0]))


def test_kappa_and_sampling_and_roundtrip(tmp_path: Path) -> None:
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("aabb"), list("bbaa")) == -1.0
    rows = [{"id": str(i), "gold": "g", "pred": "p", "verdict": v}
            for i, v in enumerate(["correct"] * 20 + ["incorrect"] * 20 + ["partial"] * 2 + ["numeric"] * 5)]
    s = stratified_sample(rows, 9)
    assert sum(r["verdict"] == "partial" for r in s) == 2 and all(r["verdict"] != "numeric" for r in s)
    judged = tmp_path / "j.json"
    judged.write_text(json.dumps({"results": rows}))
    out = tmp_path / "s.csv"
    assert export(judged, out, 9) == 9
    with out.open() as f:
        data = list(csv.DictReader(f))
    for r in data:  # a perfect human agrees with the judge
        r["human_label"] = r["judge_verdict"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, list(data[0]))
        w.writeheader()
        w.writerows(data)
    assert score(out) == {"n": 9, "agreement": 1.0, "kappa": 1.0}


def test_bandit_regret_flattens() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (600, 2))
    # arm0 best when x0>0, arm1 best otherwise
    R = np.stack([X[:, 0], -X[:, 0]], axis=1) + rng.normal(0, 0.05, (600, 2))
    res = replay(LinUCB(["a", "b"], 2, alpha=0.5), X, R)
    early = res["regret"][99]
    late = res["regret"][-1] - res["regret"][-101]
    assert late < early / 2
    assert reward(1.0, 0.2, 2.0) == pytest.approx(0.6)


def test_train_export_reload_predicts_same(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = np.where(X[:, 0] > 0.5, "agent", np.where(X[:, 1] > 0, "rag14", "rag7")).tolist()
    w = train(X, y)
    save(w, tmp_path / "w.json")
    w2 = load(tmp_path / "w.json")
    preds = [route(w2, x) for x in X]
    assert preds == [route(w, x) for x in X]
    assert np.mean(np.array(preds) == np.array(y)) > 0.7


def test_train_binary_and_tiny() -> None:
    X = np.array([[0.0], [1.0], [0.1], [0.9]])
    w = train(X, ["a", "b", "a", "b"])
    assert route(w, np.array([0.0])) == "a" and route(w, np.array([1.0])) == "b"
