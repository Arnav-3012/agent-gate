from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from engine.cost.ledger import Ledger, Pricing
from engine.tracing import schema as ev
from engine.tracing.emitter import Emitter

PRICING = Pricing(
    reference_model="m", verified_on="2026-10-21",
    usd_per_1m_tokens={"input": 0.10, "output": 0.50}, infra_usd_per_query=0.000001,
)


def stats(p: int, c: int) -> ev.LLMStats:
    return ev.LLMStats(prompt_tokens=p, completion_tokens=c, latency_s=0.1)


def test_price_is_tokens_times_rate() -> None:
    assert Ledger(PRICING).price(stats(1_000_000, 1_000_000)) == pytest.approx(0.60)


def test_cached_tokens_not_discounted() -> None:
    s = stats(1000, 0)
    s.cached_tokens = 900
    assert Ledger(PRICING).price(s) == pytest.approx(1000 * 0.10 / 1e6)


def test_emitter_stamps_event_and_stop_total() -> None:
    em = Emitter("t", [], Ledger(PRICING))
    a = em.emit(ev.Answer, text="x", llm=stats(2000, 100))
    stop = em.emit(
        ev.Stop, reason="rag_done", steps=1, total_tokens=2100, total_latency_s=0.1
    )
    assert a.llm.cost_usd == pytest.approx((2000 * 0.10 + 100 * 0.50) / 1e6)
    assert stop.total_cost_usd == pytest.approx(a.llm.cost_usd + 0.000001)


def test_no_ledger_leaves_cost_none() -> None:
    em = Emitter("t", [])
    a = em.emit(ev.Answer, text="x", llm=stats(1, 1))
    assert a.llm.cost_usd is None


def test_repo_pricing_is_verified() -> None:
    p = Pricing.from_yaml()
    assert p.reference_model != "TODO-verify" and p.usd_per_1m_tokens["input"] > 0


def test_unverified_pricing_rejected(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump({"reference_model": "x", "verified_on": None, "usd_per_1m_tokens": {}}))
    with pytest.raises(ValidationError):
        Pricing.from_yaml(f)
