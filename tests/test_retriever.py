from engine.rag.retriever import make_chunk, point_id


def test_chunk_id_deterministic_and_source_scoped() -> None:
    a = make_chunk("Paris", "Capital of France.", "squad")
    assert a.chunk_id == make_chunk("Paris", "Capital of France.", "squad").chunk_id
    assert a.chunk_id != make_chunk("Paris", "Capital of France.", "hotpotqa").chunk_id
    assert point_id(a.chunk_id) == point_id(a.chunk_id)


def test_embed_text_prepends_title() -> None:
    assert make_chunk("T", "body", "squad").embed_text == "T\nbody"
