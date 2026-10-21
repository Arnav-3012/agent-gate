"""Escalation ladder: run the cheapest arm, accept if its evidence is strong enough, else escalate."""
from collections.abc import Callable

from pydantic import BaseModel

from engine.tracing.emitter import Emitter
from engine.tracing.schema import Escalate


class Attempt(BaseModel):
    answer: str
    evidence: float  # arm-reported strength in [0,1], e.g. agent reflect confidence or top-1 retrieval score


Runner = Callable[[str], Attempt]


class CascadeResult(BaseModel):
    arm: str  # arm that produced the final answer
    attempt: Attempt
    tried: list[str]
    exhausted: bool  # no arm met the threshold; the last arm's answer is returned


def run_cascade(question: str, ladder: list[tuple[str, Runner]], threshold: float,
                emitter: Emitter | None = None) -> CascadeResult:
    """Ladder is ordered cheapest first. The last rung always answers, whatever its evidence."""
    if not ladder:
        raise ValueError("empty ladder")
    tried: list[str] = []
    for i, (name, run) in enumerate(ladder):
        attempt = run(question)
        tried.append(name)
        last = i == len(ladder) - 1
        if attempt.evidence >= threshold or last:
            return CascadeResult(arm=name, attempt=attempt, tried=tried,
                                 exhausted=last and attempt.evidence < threshold)
        if emitter:
            emitter.emit(Escalate, from_mode=name, to_mode=ladder[i + 1][0],
                         reason=f"evidence {attempt.evidence:.2f} < {threshold:.2f}")
    raise AssertionError("unreachable")
