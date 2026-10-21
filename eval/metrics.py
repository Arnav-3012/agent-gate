"""SQuAD/HotpotQA answer metrics: normalized exact match and token-level F1."""
import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return float(p == g)
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_PERIOD = re.compile(r"\b(FY\s?\d{2,4}|Q[1-4]|(?:19|20)\d{2})\b", re.IGNORECASE)


def _numbers(s: str) -> list[float]:
    """Numbers in reading order, ignoring fiscal-period tokens (FY2023, Q2, 2019) and $ , signs."""
    s = _PERIOD.sub(" ", s).replace("$", "")
    return [float(m.group(0).replace(",", "").rstrip(".")) for m in _NUM.finditer(s)]


def is_numeric_gold(gold: str) -> bool:
    """Short answers that are a number: scored by number match, everything else by the judge."""
    return bool(_numbers(gold)) and len(normalize(gold).split()) <= 3


def finance_f1(pred: str, gold: str, rel_tol: float = 0.002) -> float:
    """FinanceBench scoring: a short numeric gold answer matches only if the FIRST number in pred
    (the finance prompt puts the final answer first) is within rel_tol (0.2%); intermediate figures
    quoted later must not count. Anything else falls back to token F1."""
    g, p = _numbers(gold), _numbers(pred)
    if is_numeric_gold(gold):
        return float(bool(p) and abs(p[0] - g[0]) <= rel_tol * max(abs(g[0]), 1e-9))
    return f1(pred, gold)
