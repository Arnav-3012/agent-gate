import json
from pathlib import Path

import pytest

from eval.benchmark import benchmark, load_files


def row(i: str, f: float, cost: float, lat: float, typ: str = "bridge", lvl: str = "easy", mode: str = "rag") -> dict:
    return {"id": i, "type": typ, "level": lvl, "mode": mode, "em": float(f == 1), "f1": f,
            "cost": cost, "latency": lat}


def write(p: Path, model: str, rows: list[dict]) -> Path:
    p.write_text(json.dumps({"model": model, "results": rows}))
    return p


@pytest.fixture
def files(tmp_path: Path):
    ids = ["a", "b", "c", "d"]
    rag7 = write(tmp_path / "s7.json", "org/7b", [row(i, f, 1.0, 1.0) for i, f in zip(ids, [1, 0, 0, 1])])
    agent7 = write(tmp_path / "s7a.json", "org/7b", [row(i, f, 3.0, 4.0, mode="agent") for i, f in zip(ids, [1, 1, 0, 1])])
    rag14 = write(tmp_path / "s14.json", "org/14b", [row(i, f, 2.0, 2.0, lvl="hard") for i, f in zip(ids, [1, 0, 1, 1])])
    return load_files([rag7, agent7, rag14])


def test_arm_numbers(files) -> None:
    res = benchmark(files)["general"]
    assert res["n"] == 4
    rag = res["arms"]["7b/rag"]
    assert rag["f1"] == 0.5 and rag["em"] == 0.5 and rag["cost_per_1k"] == pytest.approx(1000.0)
    assert rag["latency_p50"] == 1.0 and rag["latency_p95"] == 1.0
    assert res["arms"]["7b/agent"]["f1"] == 0.75
    assert res["arms"]["14b/rag"]["by_level"]["hard"] == {"f1": 0.75, "n": 4}


def test_oracle_router_takes_cheapest_best(files) -> None:
    o = benchmark(files)["general"]["oracle_router"]
    # a: 7b/rag (1.0), b: 7b/agent (3.0, only one with 1), c: 14b/rag (2.0), d: 7b/rag (1.0)
    assert o["f1"] == 1.0 and o["cost_per_1k"] == pytest.approx(1750.0)
    assert o["gain_over_best_fixed"] == pytest.approx(0.25)


def test_judged_file_supersedes_raw_and_domains_split(tmp_path: Path) -> None:
    raw = write(tmp_path / "x_finance.json", "m", [row("a", 0.0, 1.0, 1.0)])
    write(tmp_path / "x_finance.judged.json", "m", [{**row("a", 0.0, 1.0, 1.0), "score": 1.0}])
    res = benchmark(load_files([raw, tmp_path / "x_finance.judged.json"]))
    assert list(res) == ["finance"] and res["finance"]["arms"]["m/rag"]["f1"] == 1.0
