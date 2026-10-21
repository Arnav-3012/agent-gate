"""Arm registry: an arm = (model server, mode). Any arm runs by name; routers only pick names."""
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, model_validator

from engine.agent.loop import AgentConfig, run_agent
from engine.llm.client import LLMClient
from engine.rag.retriever import Retriever
from engine.rag.single_pass import run_rag
from engine.tracing.emitter import Emitter
from engine.tracing.schema import Mode


class Arm(BaseModel):
    name: str
    model: str
    base_url: str
    mode: Mode
    price_multiplier: float | None = None  # None: derive from the model id via pricing.yaml (D29)

    def make_llm(self) -> LLMClient:
        return LLMClient(base_url=self.base_url, model=self.model)


class ArmRegistry(BaseModel):
    arms: list[Arm]

    @model_validator(mode="after")
    def _unique_names(self) -> "ArmRegistry":
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")
        return self

    @classmethod
    def from_yaml(cls, path: Path = Path("config/arms.yaml")) -> "ArmRegistry":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def get(self, name: str) -> Arm:
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"unknown arm {name!r}; known: {[a.name for a in self.arms]}")


class Router(Protocol):
    """Picks an arm name for a question (heuristic, classifier, bandit, cascade...)."""

    def choose(self, question: str) -> str: ...


def run_arm(arm: Arm, question: str, retriever: Retriever, llm: LLMClient, emitter: Emitter,
            agent_cfg: AgentConfig, answer_kw: dict | None = None) -> str:
    """Runs the arm's mode with the given client (callers build it from the arm, tests inject mocks)."""
    if arm.mode == "agent":
        return run_agent(question, retriever, llm, emitter, agent_cfg)
    return run_rag(question, retriever, llm, emitter, **(answer_kw or {}))
