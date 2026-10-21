"""Modelled cost: metered tokens x reference API price (config/pricing.yaml). Not a cloud bill (D4)."""
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from engine.tracing.schema import LLMStats


class Pricing(BaseModel):
    reference_model: str
    verified_on: date
    usd_per_1m_tokens: dict[str, float]
    retrieval_usd_per_query: float = 0.0
    infra_usd_per_query: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path = Path("config/pricing.yaml")) -> "Pricing":
        return cls.model_validate(yaml.safe_load(path.read_text()))  # null verified_on fails loudly


class Ledger:
    """One per query: prices each LLM call and accumulates the query total."""

    def __init__(self, pricing: Pricing) -> None:
        self.pricing = pricing
        self.llm_usd = 0.0

    def price(self, stats: LLMStats) -> float:
        p = self.pricing.usd_per_1m_tokens
        usd = (stats.prompt_tokens * p["input"] + stats.completion_tokens * p["output"]) / 1e6
        self.llm_usd += usd
        return usd

    @property
    def total_usd(self) -> float:
        return self.llm_usd + self.pricing.retrieval_usd_per_query + self.pricing.infra_usd_per_query
