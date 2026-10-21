import json
from pathlib import Path

import pytest

from engine.router.cache import SemanticCache
from engine.router.cascade import Attempt, run_cascade
from engine.router.features import extract, score_features
from engine.tracing.emitter import Emitter
from eval.benchmark import load_files
from eval.build_router_labels import build_labels


def test_features_fixed_examples() -> None:
    f = extract("Who is older, Ed Wood or Scott Derrickson?", [0.8, 0.5, 0.5])
    assert f.wh == "who" and f.has_compare and f.n_entities == 2 and f.n_digits == 0
    assert f.top1 == 0.8 and f.gap12 == pytest.approx(0.3)
    g = extract("What happened in 1999 and 2004?")
    assert g.n_digits == 8 and not g.has_compare and g.top1 == 0.0
    assert len(f.vector()) == 14


def test_entropy_is_max_for_equal_scores() -> None:
    _, _, flat = score_features([0.5, 0.5, 0.5, 0.5])
    _, _, peaked = score_features([0.9, 0.1, 0.1, 0.1])
    assert flat > peaked and score_features([]) == (0.0, 0.0, 0.0)


def _row(i: str, f: float, cost: float, mode: str) -> dict:
    return {"id": i, "type": "t", "level": "easy", "mode": mode, "em": 0.0, "f1": f, "cost": cost, "latency": 1.0}


def test_labels_cheapest_within_eps_and_missing_arms(tmp_path: Path) -> None:
    def w(name: str, model: str, rows: list[dict]) -> Path:
        (tmp_path / name).write_text(json.dumps({"model": model, "results": rows}))
        return tmp_path / name
    rag = w("a.json", "m", [_row("q1", 1, 1, "rag"), _row("q2", 0, 1, "rag"), _row("q3", 0.5, 1, "rag")])
    agent = w("b.json", "m", [_row("q1", 1, 3, "agent"), _row("q2", 1, 3, "agent"), _row("q3", 0.52, 3, "agent")])
    only = w("c.json", "x", [_row("q4", 1, 1, "rag")])  # q4 has a single arm: dropped
    arms = load_files([rag, agent, only])["general"]
    assert build_labels(arms, eps=0.05) == {"q1": "m/rag", "q2": "m/agent", "q3": "m/rag"}
    assert build_labels(arms, eps=0.0)["q3"] == "m/agent"


def _ladder(evidences: list[float]):
    calls: list[str] = []
    def rung(name: str, ev: float):
        def run(q: str) -> Attempt:
            calls.append(name)
            return Attempt(answer=f"{name}:{q}", evidence=ev)
        return name, run
    return [rung(f"arm{i}", e) for i, e in enumerate(evidences)], calls


def test_cascade_no_escalate_one_escalate_exhausted() -> None:
    ladder, calls = _ladder([0.9, 0.9])
    r = run_cascade("q", ladder, 0.5)
    assert (r.arm, r.tried, r.exhausted, calls) == ("arm0", ["arm0"], False, ["arm0"])

    ladder, _ = _ladder([0.2, 0.9, 0.9])
    em = Emitter("t", [])
    r = run_cascade("q", ladder, 0.5, em)
    assert r.arm == "arm1" and r.tried == ["arm0", "arm1"] and not r.exhausted
    assert [(e.type, e.from_mode, e.to_mode) for e in em.events] == [("escalate", "arm0", "arm1")]

    ladder, _ = _ladder([0.1, 0.2])
    r = run_cascade("q", ladder, 0.5)
    assert r.arm == "arm1" and r.exhausted


def test_cascade_rejects_empty_ladder() -> None:
    with pytest.raises(ValueError):
        run_cascade("q", [], 0.5)


VEC = {"capital of France?": [1.0, 0.0], "what is France's capital?": [0.98, 0.199],
       "who wrote Hamlet?": [0.0, 1.0]}


def test_cache_hits_near_duplicates_only_and_expires() -> None:
    now = [0.0]
    c = SemanticCache(lambda q: VEC[q], threshold=0.95, ttl_s=10, clock=lambda: now[0])
    assert c.get("capital of France?") is None
    c.put("capital of France?", "Paris", "qwen7b-rag")
    hit = c.get("what is France's capital?")
    assert hit and hit[0].answer == "Paris" and hit[0].arm == "qwen7b-rag" and hit[1] > 0.95
    assert c.get("who wrote Hamlet?") is None
    now[0] = 11.0
    assert c.get("what is France's capital?") is None
