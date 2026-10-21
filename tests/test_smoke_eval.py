import itertools

import pytest

from eval.metrics import exact_match, f1
from scripts.smoke_eval import sample, summarize, verdict


def test_metrics_normalize():
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert f1("18.7 miles", "18.7 miles away") == pytest.approx(0.8)
    assert f1("", "x") == 0.0


def test_sample_stratified_and_deterministic():
    rows = [{"id": f"{t}{i}", "type": t} for t in ("bridge", "comparison", "single") for i in range(20)]
    a, b = sample(rows, 10, 0), sample(rows, 10, 0)
    assert a == b and len(a) == 30
    assert [sum(r["type"] == t for r in a) for t in ("bridge", "comparison", "single")] == [10] * 3


def _res(mode, typ, f, cost):
    return {"mode": mode, "type": typ, "em": f, "f1": f, "tokens": 1, "cost": cost, "latency": 1.0}


def test_verdict_go_and_no_go():
    go = summarize([_res("rag", "bridge", 0.3, 1.0), _res("agent", "bridge", 0.5, 3.0)])
    assert verdict(go, 0.05)[0] and "x3.0" in verdict(go, 0.05)[1]
    no = summarize([_res("rag", "bridge", 0.4, 1.0), _res("agent", "bridge", 0.41, 3.0)])
    assert not verdict(no, 0.05)[0]


def test_finance_f1_numeric_tolerance_and_text_fallback() -> None:
    from eval.metrics import finance_f1

    assert finance_f1("1,577 million", "$1577.00") == 1.0
    assert finance_f1("about 1,600", "$1577.00") == 0.0
    assert finance_f1("Yes, commercial airplanes", "Yes. Commercial Airplanes and services") > 0.4


def test_finance_f1_first_number_after_period_tokens() -> None:
    from eval.metrics import finance_f1

    assert finance_f1("In FY2018 capex was $1,577 million", "$1577.00") == 1.0
    assert finance_f1("Q2 2019 retention is 0.21 (net earnings 0.54 of ...)", "0.54") == 0.0  # no leak
    assert finance_f1("$3,190 million in FY2020", "$3215.00") == 0.0  # 0.8% off is a wrong answer
    assert finance_f1("$381.6 million", "$382.00") == 1.0  # rounding to gold precision


def test_stable_sample_is_prefix_extendable() -> None:
    rows = [{"id": f"{t}{i}", "type": t} for t in ("a", "b") for i in range(50)]
    small, big = sample(rows, 5, 0, stable=True), sample(rows, 20, 0, stable=True)
    assert [q["id"] for q in small if q["type"] == "a"] == [q["id"] for q in big if q["type"] == "a"][:5]


def test_windows_cover_text_and_break_on_spaces() -> None:
    from scripts.ingest_finance import squash, windows

    text = squash("Revenue 1,577 " * 400)
    spans = windows(text, size=900, overlap=150)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(text[a:b].split()[-1] in ("Revenue", "1,577") for a, b in spans)  # no cut numbers
    assert all(b - a <= 900 for a, b in spans) and all(s2[0] < s1[1] for s1, s2 in itertools.pairwise(spans))


def test_excluding_a_type_after_sampling_keeps_other_types_identical() -> None:
    rows = [{"id": f"{t}{i}", "type": t} for t in ("a", "m", "n") for i in range(30)]
    full = sample(rows, 10, 0, stable=True)
    kept = [q for q in full if q["type"] != "m"]
    assert [q["id"] for q in kept if q["type"] == "n"] == [q["id"] for q in full if q["type"] == "n"]
    # removing the type BEFORE sampling would shift the seeded stream and change type "n"
    assert [q["id"] for q in sample([r for r in rows if r["type"] != "m"], 10, 0, stable=True)
            if q["type"] == "n"] != [q["id"] for q in full if q["type"] == "n"]
