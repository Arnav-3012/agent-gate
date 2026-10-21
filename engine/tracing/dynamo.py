"""DynamoDB trace sink (TRACE_SINK=dynamodb|both). Item layout is read by infra/lambdas/store.py:
pk trace_id (S), sk seq (N), `event` = the event JSON, `ttl` = epoch seconds (table TTL attribute).
The event is stored as a JSON string because DynamoDB rejects Python floats."""
import os
import time
from pathlib import Path
from typing import Any

from engine.tracing.emitter import JsonlSink, Sink
from engine.tracing.schema import TraceEvent, event_adapter

TTL_S = 7 * 24 * 3600


class DynamoSink:
    def __init__(self, table: Any, ttl_s: int = TTL_S, clock: Any = time.time) -> None:
        self.table = table  # boto3 Table (or anything with put_item)
        self.ttl_s = ttl_s
        self.clock = clock

    def __call__(self, event: TraceEvent) -> None:
        self.table.put_item(Item={
            "trace_id": event.trace_id,
            "seq": event.seq,
            "type": event.type,
            "event": event_adapter.dump_json(event).decode(),
            "ttl": int(self.clock()) + self.ttl_s,
        })


def boto_table(name: str | None = None, region: str | None = None) -> Any:
    import boto3  # deferred: only needed when the dynamodb sink is on

    return boto3.resource(
        "dynamodb", region_name=region or os.environ.get("AWS_REGION", "ap-south-1")
    ).Table(name or os.environ.get("TRACE_TABLE", "agentgate-traces"))


def make_sinks(jsonl_path: Path, kind: str | None = None, table: Any = None) -> list[Sink]:
    """`kind` defaults to $TRACE_SINK. The local JSONL file stays the source for GET /trace on
    the engine, so `dynamodb` alone is for a headless engine behind the AWS API."""
    kind = kind or os.environ.get("TRACE_SINK", "jsonl")
    if kind not in {"jsonl", "dynamodb", "both"}:
        raise ValueError(f"TRACE_SINK must be jsonl|dynamodb|both, got {kind!r}")
    sinks: list[Sink] = []
    if kind in {"jsonl", "both"}:
        sinks.append(JsonlSink(jsonl_path))
    if kind in {"dynamodb", "both"}:
        sinks.append(DynamoSink(table or boto_table()))
    return sinks
