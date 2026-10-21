"""Export a finished trace as OTLP/JSON (resourceSpans) using the OpenTelemetry GenAI conventions.

No SDK dependency: the dict can be POSTed to any OTLP/HTTP collector (/v1/traces) or loaded by
Jaeger/Tempo/Langfuse-style backends. Our own TraceEvent schema stays the source of truth.
"""
import hashlib

from engine.tracing import schema as ev

_NS = 1_000_000_000


def _span_id(trace_id: str, seq: int) -> str:
    return hashlib.sha1(f"{trace_id}:{seq}".encode()).hexdigest()[:16]


def _attr(key: str, value: object) -> dict:
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _event_span(e: ev.TraceEvent, root: str, model: str) -> dict:
    attrs = {"agentgate.event": e.type}
    latency = 0.0
    llm = getattr(e, "llm", None)
    if llm is not None:
        latency = llm.latency_s
        attrs |= {"gen_ai.operation.name": "chat", "gen_ai.request.model": model,
                  "gen_ai.usage.input_tokens": llm.prompt_tokens,
                  "gen_ai.usage.output_tokens": llm.completion_tokens}
        if llm.cost_usd is not None:
            attrs["agentgate.cost_usd"] = llm.cost_usd
    if isinstance(e, ev.Retrieve):
        latency = e.latency_s
        attrs |= {"agentgate.query": e.query, "agentgate.hits": len(e.hits)}
    if isinstance(e, ev.Reflect):
        attrs |= {"agentgate.confidence": e.confidence, "agentgate.new_evidence": e.new_evidence}
    if isinstance(e, ev.RouteDecided):
        attrs |= {"agentgate.route": e.route, "agentgate.reason": e.reason}
    end = int(e.ts * _NS)
    return {
        "traceId": e.trace_id, "spanId": _span_id(e.trace_id, e.seq), "parentSpanId": root,
        "name": e.type, "kind": 1, "startTimeUnixNano": str(end - int(latency * _NS)),
        "endTimeUnixNano": str(end), "attributes": [_attr(k, v) for k, v in attrs.items()],
    }


def to_otlp(events: list[ev.TraceEvent], model: str = "unknown") -> dict:
    """One root span `agentgate.query` (from first event's start to the stop) + one child per event."""
    first, last = events[0], events[-1]
    root = _span_id(first.trace_id, -1)
    stop = last if isinstance(last, ev.Stop) else None
    root_attrs = {"gen_ai.request.model": model}
    if stop:
        root_attrs |= {"agentgate.stop_reason": stop.reason, "agentgate.steps": stop.steps,
                       "gen_ai.usage.total_tokens": stop.total_tokens}
        if stop.total_cost_usd is not None:
            root_attrs["agentgate.cost_usd"] = stop.total_cost_usd
    start = int(first.ts * _NS) - (int(stop.total_latency_s * _NS) if stop else 0)
    root_span = {
        "traceId": first.trace_id, "spanId": root, "name": "agentgate.query", "kind": 2,
        "startTimeUnixNano": str(start), "endTimeUnixNano": str(int(last.ts * _NS)),
        "attributes": [_attr(k, v) for k, v in root_attrs.items()],
        "status": {"code": 2 if stop and stop.reason == "error" else 1},
    }
    spans = [root_span] + [_event_span(e, root, model) for e in events]
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", "agentgate-engine")]},
        "scopeSpans": [{"scope": {"name": "agentgate.tracing"}, "spans": spans}],
    }]}
