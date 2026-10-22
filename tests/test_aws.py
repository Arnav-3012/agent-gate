import json
import sys
from pathlib import Path

import pytest

from engine.tracing.dynamo import DynamoSink, make_sinks
from engine.tracing.emitter import JsonlSink
from engine.tracing.schema import Answer, LLMStats, Stop
from tests.test_api import make_client

sys.path.insert(0, str(Path(__file__).parent.parent / "infra" / "lambdas"))
import get_trace as trace_fn
import post_query as router
from store import TraceStore

TID = "a" * 32


class FakeTable:
    """put_item + query for one partition key, like the slice of boto3 Table the code uses."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def put_item(self, Item: dict) -> None:
        self.items.append(Item)

    def query(self, KeyConditionExpression, ExclusiveStartKey=None) -> dict:
        wanted = KeyConditionExpression.get_expression()["values"][1]
        hits = [i for i in self.items if i["trace_id"] == wanted]
        if ExclusiveStartKey is None and len(hits) > 1:  # force a second page
            return {"Items": hits[:1], "LastEvaluatedKey": {"seq": hits[0]["seq"]}}
        return {"Items": hits[1:] if ExclusiveStartKey else hits}


def events(table: FakeTable, n: int = 2) -> None:
    sink = DynamoSink(table, clock=lambda: 1000.0)
    llm = LLMStats(prompt_tokens=1, completion_tokens=1, cached_tokens=0, latency_s=0.1)
    for seq in range(n - 1):
        sink(Answer(trace_id=TID, seq=seq, ts=1.0, text="x", llm=llm))
    sink(Stop(trace_id=TID, seq=n - 1, ts=2.0, reason="rag_done", steps=1, total_tokens=2,
              total_cost_usd=0.0, total_latency_s=0.1, error=None))


def test_sink_item_layout_and_ttl() -> None:
    t = FakeTable()
    events(t)
    item = t.items[0]
    assert item["trace_id"] == TID and item["seq"] == 0 and item["ttl"] == 1000 + 7 * 24 * 3600
    assert json.loads(item["event"])["type"] == "answer"


def test_trace_lambda_reads_what_the_sink_wrote_with_pagination_and_after() -> None:
    t = FakeTable()
    events(t, n=3)
    store = TraceStore(t)
    q = lambda **p: json.loads(trace_fn.handler(
        {"pathParameters": {"trace_id": TID}, "queryStringParameters": p or None}, store=store)["body"])
    body = q()
    assert [e["seq"] for e in body["events"]] == [0, 1, 2] and body["done"]
    assert [e["seq"] for e in q(after="1")["events"]] == [2]
    assert q(after="2")["events"] == [] and q(after="2")["done"]


def test_trace_lambda_pending_bad_id_bad_after() -> None:
    store = TraceStore(FakeTable())
    ok = trace_fn.handler({"pathParameters": {"trace_id": TID}}, store=store)
    assert json.loads(ok["body"]) == {"trace_id": TID, "events": [], "done": False}
    assert trace_fn.handler({"pathParameters": {"trace_id": "../x"}}, store=store)["statusCode"] == 404
    bad = trace_fn.handler(
        {"pathParameters": {"trace_id": TID}, "queryStringParameters": {"after": "x"}}, store=store)
    assert bad["statusCode"] == 422


def call_router(body: object, post, monkeypatch) -> dict:
    monkeypatch.setenv("ENGINE_URL", "http://engine:8000/")
    return router.handler({"body": json.dumps(body)}, post=post)


def test_router_mints_trace_id_and_forwards(monkeypatch) -> None:
    sent: list = []
    res = call_router({"question": "q?", "arm": "qwen7b-rag"}, lambda u, b: sent.append((u, b)) or 202,
                      monkeypatch)
    tid = json.loads(res["body"])["trace_id"]
    assert res["statusCode"] == 202 and len(tid) == 32
    assert sent == [("http://engine:8000/run", {"question": "q?", "trace_id": tid, "arm": "qwen7b-rag"})]


@pytest.mark.parametrize("body,post,status", [
    ({"question": "  "}, lambda u, b: 202, 422),
    ({"nope": 1}, lambda u, b: 202, 422),
    ({"question": "q", "arm": "zzz"}, lambda u, b: 422, 422),
    ({"question": "q"}, lambda u, b: 500, 502),
])
def test_router_rejections(body, post, status, monkeypatch) -> None:
    assert call_router(body, post, monkeypatch)["statusCode"] == status


def test_health_route_proxies_engine(monkeypatch) -> None:
    monkeypatch.setenv("ENGINE_URL", "http://engine:8000/")
    ev = {"routeKey": "GET /health"}
    ok = router.handler(ev, get=lambda u: (200, {"ok": True, "llm": True, "qdrant": True}) if u.endswith("/health") else (404, None))
    assert ok["statusCode"] == 200 and json.loads(ok["body"])["llm"] is True
    assert router.handler(ev, get=lambda u: (500, None))["statusCode"] == 502

    def down(url: str) -> tuple[int, None]:
        raise ConnectionRefusedError

    assert router.handler(ev, get=down)["statusCode"] == 502


def test_router_engine_unreachable(monkeypatch) -> None:
    def down(url: str, body: dict) -> int:
        raise ConnectionRefusedError

    assert call_router({"question": "q"}, down, monkeypatch)["statusCode"] == 502
    assert router.handler({"body": "{not json"})["statusCode"] == 422


def test_make_sinks(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    assert [type(s) for s in make_sinks(p, "jsonl")] == [JsonlSink]
    assert [type(s) for s in make_sinks(p, "both", table=FakeTable())] == [JsonlSink, DynamoSink]
    assert [type(s) for s in make_sinks(p, "dynamodb", table=FakeTable())] == [DynamoSink]
    with pytest.raises(ValueError):
        make_sinks(p, "s3")


def test_run_requires_shared_key_when_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENGINE_KEY", "s3cret")
    with make_client(tmp_path) as c:
        assert c.post("/run", json={"question": "q"}).status_code == 401
        assert c.post("/run", json={"question": "q"}, headers={"x-agentgate-key": "no"}).status_code == 401
        assert c.post("/run", json={"question": "q"}, headers={"x-agentgate-key": "s3cret"}).status_code == 202
        assert c.get("/health").status_code == 200  # only /run is guarded


def test_router_sends_shared_key(monkeypatch) -> None:
    seen: dict = {}

    class R:
        status = 202
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_open(req, timeout):
        seen.update(req.headers)
        return R()

    monkeypatch.setenv("ENGINE_KEY", "s3cret")
    monkeypatch.setattr(router.urllib.request, "urlopen", fake_open)
    assert router.post_json("http://e/run", {"question": "q"}) == 202
    assert seen["X-agentgate-key"] == "s3cret"


def test_run_uses_router_minted_trace_id(tmp_path: Path) -> None:
    with make_client(tmp_path) as c:
        assert c.post("/run", json={"question": "q", "trace_id": TID}).json() == {"trace_id": TID}
        assert c.get(f"/trace/{TID}").json()["done"]
        assert c.post("/run", json={"question": "q", "trace_id": "bad"}).status_code == 422


def test_stack_synthesizes_expected_resources(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("aws_cdk")  # infra dependency group; CI installs only the default groups
    from aws_cdk import App, assertions

    sys.path.insert(0, str(Path(__file__).parent.parent / "infra"))
    from stack import AgentGateStack

    stack = AgentGateStack(App(outdir=str(tmp_path)), "T", engine_url="https://e.example", budget_email="a@b.c", with_cdn=True)
    tpl = assertions.Template.from_stack(stack)
    tpl.has_resource_properties("AWS::DynamoDB::Table", {
        "TimeToLiveSpecification": {"AttributeName": "ttl", "Enabled": True},
        "KeySchema": [{"AttributeName": "trace_id", "KeyType": "HASH"}, {"AttributeName": "seq", "KeyType": "RANGE"}],
    })
    tpl.has_resource_properties("AWS::ApiGatewayV2::Stage", {
        "StageName": "$default", "DefaultRouteSettings": {"ThrottlingRateLimit": 10, "ThrottlingBurstLimit": 20}})
    tpl.resource_count_is("AWS::CloudFront::Function", 1)
    tpl.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "GET /health"})
    tpl.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "POST /query"})
    tpl.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "GET /trace/{trace_id}"})
    tpl.has_resource_properties("AWS::Lambda::Function", {
        "Handler": "post_query.handler", "Environment": {"Variables": {"ENGINE_URL": "https://e.example"}}})
    tpl.has_resource_properties("AWS::Budgets::Budget", {"Budget": assertions.Match.object_like(
        {"BudgetLimit": {"Amount": 5, "Unit": "USD"}})})
    tpl.resource_count_is("AWS::CloudFront::Distribution", 1)


def test_stack_without_cdn_has_no_cloudfront_or_bucket(tmp_path: Path) -> None:
    pytest.importorskip("aws_cdk")
    from aws_cdk import App, assertions

    sys.path.insert(0, str(Path(__file__).parent.parent / "infra"))
    from stack import AgentGateStack

    stack = AgentGateStack(App(outdir=str(tmp_path)), "T2", engine_url="https://e.example",
                           budget_email="a@b.c", with_cdn=False)
    tpl = assertions.Template.from_stack(stack)
    for t in ("AWS::CloudFront::Distribution", "AWS::CloudFront::Function", "AWS::S3::Bucket"):
        tpl.resource_count_is(t, 0)
    tpl.resource_count_is("AWS::ApiGatewayV2::Api", 1)
