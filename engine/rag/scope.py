"""Metadata scope for filing questions: company + fiscal year parsed from the question text.

FinanceBench questions name the company and usually the year ("FY2018 ... for 3M"), and every chunk
title is `{company} {period} {doc_type} p.N`. Parsing both lets retrieval ignore other filings
(D43) without re-ingesting or adding payload fields. Companies are read from the corpus itself.
"""
import re
from collections.abc import Iterable
from typing import NamedTuple

from engine.rag.retriever import Hit

TITLE = re.compile(r"^(?P<company>.+) (?P<period>\d{4}) (?P<doc_type>\S+) p\.\d+$")
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


class Scope(NamedTuple):
    companies: frozenset[str]
    years: frozenset[int]

    def __bool__(self) -> bool:
        return bool(self.companies)


def title_parts(title: str) -> tuple[str, int] | None:
    """(company, period) from a filing-chunk title, None for titles of other corpora."""
    m = TITLE.match(title)
    return (m["company"], int(m["period"])) if m else None


def companies_in(titles: Iterable[str]) -> list[str]:
    """Distinct company names, longest first so 'American Express' wins over any shorter prefix."""
    parts = (title_parts(t) for t in titles)
    return sorted({p[0] for p in parts if p}, key=lambda c: (-len(c), c))


# Nicknames the questions use for a filing company (ticker-style, not derivable from the title).
ALIASES = {"JPM": "JPMorgan", "JnJ": "Johnson & Johnson", "AMEX": "American Express"}


def _squash(text: str) -> str:
    return re.sub(r"[\s-]+", " ", text)


def _names(company: str, companies: Iterable[str]) -> list[str]:
    """Surface forms of a company: its title name, ticker nicknames, and its first word when that
    is unique in the corpus ('Ulta' for 'Ulta Beauty', but not 'American')."""
    names = [company, *(a for a, c in ALIASES.items() if c == company)]
    first = company.split()[0]
    if " " in company and len(first) >= 3 and sum(c.split()[0] == first for c in companies) == 1:
        names.append(first)
    return names


def _mentions(question: str, company: str, companies: Iterable[str]) -> bool:
    """Case-insensitive, hyphen-insensitive, but a lowercase hit does not count: 'block' in
    'building block' is not the company Block, while 'Pepsico' / 'AMCOR' are."""
    text = _squash(question)
    for name in _names(company, companies):
        pattern = rf"(?<![\w-]){re.escape(_squash(name))}(?![\w-])"
        if any(m.group() != m.group().lower() for m in re.finditer(pattern, text, re.IGNORECASE)):
            return True
    return False


def parse_scope(question: str, companies: Iterable[str]) -> Scope:
    """Companies named in the question (all of them, for cross-company questions) and every year
    it mentions. Empty companies means 'no scope': the caller keeps the unfiltered ranking."""
    companies = list(companies)
    found = frozenset(c for c in companies if _mentions(question, c, companies))
    return Scope(found, frozenset(int(y) for y in YEAR.findall(question)))


def apply_scope(hits: list[Hit], scope: Scope, k: int) -> list[Hit]:
    """Top-k hits inside the scope. Year is a preference: if no hit of a named company matches a
    named year (e.g. the question says 'since 2015' but only a 2022 filing exists), keep the
    company's hits. No company hit at all -> the unfiltered top-k (never return nothing)."""
    if not scope:
        return hits[:k]
    parts = [(h, title_parts(h.chunk.title)) for h in hits]
    of_company = [(h, p) for h, p in parts if p and p[0] in scope.companies]
    in_year = [(h, p) for h, p in of_company if p[1] in scope.years]
    keep = in_year if scope.years and in_year else of_company
    return [h for h, _ in keep][:k] or hits[:k]
