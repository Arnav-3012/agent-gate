"""GET /trace/{trace_id}?after=N -> same body as the engine's GET /trace (TraceView)."""
import json
import os
import re
from typing import Any

from store import TraceStore

TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_store: TraceStore | None = None


def response(status: int, body: Any) -> dict[str, Any]:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def get_store() -> TraceStore:
    global _store
    if _store is None:
        import boto3

        _store = TraceStore(boto3.resource("dynamodb").Table(os.environ["TRACE_TABLE"]))
    return _store


def handler(event: dict[str, Any], context: Any = None, store: TraceStore | None = None) -> dict[str, Any]:
    trace_id = (event.get("pathParameters") or {}).get("trace_id", "")
    if not TRACE_ID.match(trace_id):
        return response(404, {"detail": "unknown trace"})
    try:
        after = int((event.get("queryStringParameters") or {}).get("after", -1))
    except ValueError:
        return response(422, {"detail": "after must be an integer"})
    events = (store or get_store()).events(trace_id)
    # An id with no items yet is indistinguishable from an unknown one (the engine writes the
    # first event after /run is accepted), so it reads as pending: the poller keeps polling.
    return response(200, {
        "trace_id": trace_id,
        "events": [e for e in events if e["seq"] > after],
        "done": any(e["type"] == "stop" for e in events),
    })
