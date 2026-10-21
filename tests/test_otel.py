from engine.tracing import schema as ev
from engine.tracing.emitter import Emitter
from engine.tracing.otel import to_otlp


def test_otlp_shape_and_genai_attrs() -> None:
    em = Emitter("a" * 32, [])
    llm = ev.LLMStats(prompt_tokens=10, completion_tokens=2, latency_s=0.5, cost_usd=1e-6)
    em.emit(ev.Retrieve, step=0, query="q", hits=[], latency_s=0.1)
    em.emit(ev.Answer, text="x", llm=llm)
    em.emit(ev.Stop, reason="rag_done", steps=0, total_tokens=12, total_latency_s=0.6)
    spans = to_otlp(em.events, "qwen")["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["agentgate.query", "retrieve", "answer", "stop"]
    assert all(s["traceId"] == "a" * 32 and len(s["spanId"]) == 16 for s in spans)
    assert all(s["parentSpanId"] == spans[0]["spanId"] for s in spans[1:])
    keys = {a["key"] for a in spans[2]["attributes"]}
    assert {"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.request.model"} <= keys
