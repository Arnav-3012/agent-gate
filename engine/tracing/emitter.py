"""Builds sequenced trace events and fans them out to sinks (JSONL file, DynamoDB later)."""
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from engine.cost.ledger import Ledger
from engine.tracing.schema import Stop, TraceEvent, event_adapter

Sink = Callable[[TraceEvent], None]


class JsonlSink:
    """Append-only, one event per line; the same files feed /replay and the dashboard replay mode."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: TraceEvent) -> None:
        with self.path.open("a") as f:
            f.write(event_adapter.dump_json(event).decode() + "\n")


def read_jsonl(path: Path) -> list[TraceEvent]:
    return [event_adapter.validate_json(line) for line in path.read_text().splitlines() if line]


class Emitter:
    """One per query. emit(Cls, **fields) stamps trace_id/seq/ts (and cost, if a ledger is set),
    validates, then calls every sink."""

    def __init__(self, trace_id: str, sinks: list[Sink], ledger: Ledger | None = None) -> None:
        self.trace_id = trace_id
        self.sinks = sinks
        self.ledger = ledger
        self.events: list[TraceEvent] = []

    def emit(self, cls: type[Any], **fields: Any) -> TraceEvent:
        event = cls(trace_id=self.trace_id, seq=len(self.events), ts=time.time(), **fields)
        if self.ledger:
            llm = getattr(event, "llm", None)
            if llm is not None:
                llm.cost_usd = self.ledger.price(llm)
            if isinstance(event, Stop):
                event.total_cost_usd = self.ledger.total_usd
        self.events.append(event)
        for sink in self.sinks:
            sink(event)
        return event
