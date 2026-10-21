from engine.rag.retriever import Hit, make_chunk
from engine.rag.scope import apply_scope, companies_in, parse_scope, title_parts

TITLES = ["3M 2022 10k p.4", "3M 2018 10k p.9", "American Express 2022 10k p.1", "Block 2020 10k p.2",
          "Amcor 2023 Earnings p.3", "Coca-Cola 2021 10k p.7"]
COS = companies_in(TITLES)


def hit(title: str, score: float = 0.5) -> Hit:
    return Hit(chunk=make_chunk(title, title, "financebench"), score=score)


def test_title_parts() -> None:
    assert title_parts("American Express 2022 10k p.12") == ("American Express", 2022)
    assert title_parts("Amcor 2023 Earnings p.3") == ("Amcor", 2023)
    assert title_parts("Paris") is None


def test_companies_longest_first_and_distinct() -> None:
    assert COS[0] == "American Express" and COS.count("3M") == 1 and "Paris" not in COS


def test_parse_company_and_years() -> None:
    s = parse_scope("What is the FY2018 capex for 3M? Compare with FY2019 and Q2 of 2022.", COS)
    assert s.companies == {"3M"} and s.years == {2018, 2019, 2022}


def test_single_word_company_needs_capitalisation() -> None:
    assert not parse_scope("Is this a building block of growth in 2020?", COS)
    assert parse_scope("Did Block grow in 2020?", COS).companies == {"Block"}


def test_case_hyphen_and_alias_forms() -> None:
    cos = companies_in(TITLES + ["Ulta Beauty 2023 Earnings p.2", "American Water Works 2020 10k p.1",
                                 "JPMorgan 2022 10k p.3", "Amcor 2023 10k p.1"])
    assert parse_scope("Does AMCOR grow?", cos).companies == {"Amcor"}
    assert parse_scope("ROA for Coca Cola?", cos).companies == {"Coca-Cola"}
    assert parse_scope("Is JPM profitable?", cos).companies == {"JPMorgan"}
    assert parse_scope("Ulta margins", cos).companies == {"Ulta Beauty"}
    assert not parse_scope("American utilities", cos)  # first word shared by two companies


def test_multi_company_question() -> None:
    assert parse_scope("Is Coca-Cola bigger than 3M?", COS).companies == {"3M", "Coca-Cola"}


def test_no_company_means_no_scope() -> None:
    s = parse_scope("What happened in 2020?", COS)
    assert not s and apply_scope([hit("3M 2022 10k p.4")], s, 5)[0].chunk.title == "3M 2022 10k p.4"


def test_apply_scope_filters_company_and_year() -> None:
    hits = [hit("Block 2020 10k p.2"), hit("3M 2022 10k p.4"), hit("3M 2018 10k p.9")]
    s = parse_scope("3M FY2018 capex", COS)
    assert [h.chunk.title for h in apply_scope(hits, s, 5)] == ["3M 2018 10k p.9"]


def test_year_is_a_preference_not_a_requirement() -> None:
    hits = [hit("Block 2020 10k p.2"), hit("3M 2022 10k p.4")]
    s = parse_scope("3M since 2010", COS)
    assert [h.chunk.title for h in apply_scope(hits, s, 5)] == ["3M 2022 10k p.4"]


def test_company_absent_from_pool_falls_back_to_unfiltered() -> None:
    hits = [hit("Block 2020 10k p.2")]
    assert apply_scope(hits, parse_scope("3M FY2018", COS), 5) == hits


def test_top_k_cut_after_filter() -> None:
    hits = [hit(f"3M 2022 10k p.{i}") for i in range(10)]
    assert len(apply_scope(hits, parse_scope("3M 2022", COS), 8)) == 8
