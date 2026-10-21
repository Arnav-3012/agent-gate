"""Query features for routers (no LLM): surface cues from the question + retrieval-score shape."""
import math
import re

from pydantic import BaseModel

WH = ("who", "what", "when", "where", "which", "why", "how")
COMPARE = re.compile(
    r"\b(both|same|older|younger|larger|smaller|more|less|than|compare|difference|versus|vs)\b", re.IGNORECASE)
ENTITY = re.compile(r"(?<!^)(?<![.?!] )\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*")


class Features(BaseModel):
    n_words: int
    wh: str  # first wh-word found, "" if none
    n_entities: int
    has_compare: bool
    n_digits: int
    top1: float = 0.0
    gap12: float = 0.0
    entropy: float = 0.0  # softmax entropy of the scores (high: no clear winner)

    def vector(self) -> list[float]:
        wh = [float(self.wh == w) for w in WH]
        return [self.n_words, self.n_entities, float(self.has_compare), self.n_digits,
                self.top1, self.gap12, self.entropy, *wh]


def score_features(scores: list[float]) -> tuple[float, float, float]:
    """(top-1 score, top1-top2 gap, entropy of softmax over the scores); zeros when no hits."""
    if not scores:
        return 0.0, 0.0, 0.0
    s = sorted(scores, reverse=True)
    exp = [math.exp(x) for x in s]
    z = sum(exp)
    entropy = -sum(p * math.log(p) for p in (e / z for e in exp))
    return s[0], s[0] - s[1] if len(s) > 1 else s[0], entropy


def extract(question: str, scores: list[float] | None = None) -> Features:
    words = question.split()
    lower = [w.lower().strip("?,.") for w in words]
    top1, gap, ent = score_features(scores or [])
    return Features(
        n_words=len(words),
        wh=next((w for w in lower if w in WH), ""),
        n_entities=len(ENTITY.findall(question)),
        has_compare=bool(COMPARE.search(question)),
        n_digits=sum(c.isdigit() for c in question),
        top1=top1, gap12=gap, entropy=ent,
    )
