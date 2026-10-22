"""POST /query {question, arm?, mode?} -> 202 {trace_id}. Mints the trace id, tells the engine
(ENGINE_URL/run answers 202 at once and works in the background), returns. The browser then polls
GET /trace/{id}, which reads what the engine writes to DynamoDB."""
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

ENGINE_TIMEOUT_S = 5.0


def response(status: int, body: Any) -> dict[str, Any]:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def post_json(url: str, body: dict[str, Any], timeout: float = ENGINE_TIMEOUT_S) -> int:
    headers = {"content-type": "application/json"}
    if key := os.environ.get("ENGINE_KEY"):  # shared secret checked by the engine's POST /run
        headers["x-agentgate-key"] = key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def get_json(url: str, timeout: float = ENGINE_TIMEOUT_S) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def health(get: Callable[[str], tuple[int, Any]] = get_json) -> dict[str, Any]:
    """GET /health: the engine's own answer, so the dashboard's live/replay decision sees the real stack."""
    try:
        status, body = get(os.environ["ENGINE_URL"].rstrip("/") + "/health")
    except (OSError, urllib.error.URLError, ValueError):
        return response(502, {"detail": "engine unreachable"})
    return response(200, body) if status == 200 and isinstance(body, dict) else response(502, {"detail": "engine unhealthy"})


def handler(
    event: dict[str, Any], context: Any = None,
    post: Callable[[str, dict[str, Any]], int] = post_json,
    get: Callable[[str], tuple[int, Any]] = get_json,
) -> dict[str, Any]:
    if event.get("routeKey") == "GET /health":
        return health(get)
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return response(422, {"detail": "body must be JSON"})
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return response(422, {"detail": "question is required"})
    run: dict[str, Any] = {"question": question, "trace_id": uuid.uuid4().hex}
    for key in ("arm", "mode", "domain"):  # validated by the engine; its 422 is passed through
        if body.get(key) is not None:
            run[key] = body[key]
    try:
        status = post(os.environ["ENGINE_URL"].rstrip("/") + "/run", run)
    except (OSError, urllib.error.URLError):
        return response(502, {"detail": "engine unreachable"})
    if status == 202:
        return response(202, {"trace_id": run["trace_id"]})
    return response(422 if status == 422 else 502, {"detail": f"engine answered {status}"})
