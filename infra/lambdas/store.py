"""Read side of the DynamoDB trace table written by engine/tracing/dynamo.py (same item layout).
Stdlib + boto3 only: the Lambda runtime has no pydantic or httpx."""
import json
from typing import Any


class TraceStore:
    def __init__(self, table: Any) -> None:
        self.table = table  # boto3 Table (or a fake with query())

    def events(self, trace_id: str) -> list[dict[str, Any]]:
        """All events of a trace in seq order. A trace is a handful of items, so no `after` key
        condition: the handler filters, and `done` sees the stop even if the poller already has it."""
        from boto3.dynamodb.conditions import Key

        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"KeyConditionExpression": Key("trace_id").eq(trace_id)}
        while True:
            page = self.table.query(**kwargs)
            items += page["Items"]
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        return [json.loads(i["event"]) for i in sorted(items, key=lambda i: int(i["seq"]))]
