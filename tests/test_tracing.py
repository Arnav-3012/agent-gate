import re
from pathlib import Path

from engine.tracing.emitter import Emitter, JsonlSink, read_jsonl
from engine.tracing.schema import (
    SCHEMA_VERSION,
    Answer,
    Hit,
    LLMStats,
    Retrieve,
    RouteDecided,
    Stop,
    event_adapter,
)

LLM = LLMStats(prompt_tokens=10, completion_tokens=2, latency_s=0.1)


def test_jsonl_roundtrip_and_seq(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    em = Emitter("t1", [JsonlSink(path)])
    em.emit(RouteDecided, route="rag", p_agent=0.2, features={"len": 7.0})
    em.emit(Retrieve, step=0, query="q", hits=[Hit(chunk_id="c", title="T", score=0.9)], latency_s=0.05)
    em.emit(Answer, text="a", llm=LLM)
    em.emit(Stop, reason="rag_done", steps=1, total_tokens=12, total_latency_s=0.2)
    events = read_jsonl(path)
    assert events == em.events
    assert [e.seq for e in events] == [0, 1, 2, 3]
    assert [e.type for e in events][-1] == "stop"


def test_discriminator_rejects_unknown_type() -> None:
    bad = '{"type": "nope", "trace_id": "t", "seq": 0, "ts": 0}'
    try:
        event_adapter.validate_json(bad)
    except ValueError:
        return
    raise AssertionError("unknown event type accepted")


def test_ts_mirror_matches_python() -> None:
    ts = Path("dashboard/lib/schema.ts").read_text()
    ts_types = set(re.findall(r'type: "(\w+)"', ts))
    py_types = set(re.findall(r'type: Literal\["(\w+)"\]', Path("engine/tracing/schema.py").read_text()))
    assert ts_types == py_types
    assert f"SCHEMA_VERSION = {SCHEMA_VERSION}" in ts


def test_v2_route_and_escalate_still_validate() -> None:
    base = {"trace_id": "t", "seq": 0, "ts": 0}

    def parse(**fields):
        return event_adapter.validate_python({**base, **fields})

    r = parse(type="route_decided", route="rag")
    e = parse(type="escalate", from_mode="rag", to_mode="agent", reason="x")
    assert r.route == "rag" and e.to_mode == "agent"
    assert parse(type="route_decided", route="qwen14b-agent").route == "qwen14b-agent"
