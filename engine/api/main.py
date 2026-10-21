"""HTTP face of the engine: POST /run starts a traced query, GET /trace/{id} polls its events."""
import re
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from engine.agent.loop import AgentConfig, run_agent
from engine.cost.ledger import Ledger, Pricing
from engine.llm.client import LLMClient
from engine.rag.retriever import COLLECTION, Retriever
from engine.rag.single_pass import run_rag
from engine.tracing.emitter import Emitter, JsonlSink
from engine.tracing.schema import TraceEvent, event_adapter

TRACE_ID = re.compile(r"^[0-9a-f]{32}$")  # also keeps ids from escaping trace_dir


class RunRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["rag", "agent"] = "rag"


class RunAccepted(BaseModel):
    trace_id: str


class TraceView(BaseModel):
    trace_id: str
    events: list[TraceEvent]
    done: bool  # a stop event has been written


class Engine:
    """Everything a query needs. Built once; the LLM client is per query (its meter is per query)."""

    def __init__(
        self,
        retriever: Retriever,
        make_llm: Callable[[], LLMClient] = LLMClient,
        trace_dir: Path = Path("traces"),
        pricing: Pricing | None = None,
        agent_cfg: AgentConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.make_llm = make_llm
        self.trace_dir = trace_dir
        self.pricing = pricing or Pricing.from_yaml()
        self.agent_cfg = agent_cfg or AgentConfig.from_yaml()

    def path(self, trace_id: str) -> Path:
        return self.trace_dir / f"{trace_id}.jsonl"

    def run(self, trace_id: str, req: RunRequest) -> None:
        """Runs one query to completion; run_rag/run_agent always end the trace with a stop."""
        llm = self.make_llm()
        emitter = Emitter(trace_id, [JsonlSink(self.path(trace_id))], Ledger(self.pricing, llm.model))
        try:
            if req.mode == "rag":
                run_rag(req.question, self.retriever, llm, emitter)
            else:
                run_agent(req.question, self.retriever, llm, emitter, self.agent_cfg)
        finally:
            llm.close()


def read_events(path: Path) -> list[TraceEvent]:
    """Complete lines only: a poll can land mid-write of the last line."""
    lines = path.read_text().splitlines(keepends=True)
    return [event_adapter.validate_json(l) for l in lines if l.endswith("\n")]


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine or Engine(Retriever())  # loads the embedder: slow, so once
        yield

    app = FastAPI(title="AgentGate engine", lifespan=lifespan)

    def eng(request: Request) -> Engine:
        return request.app.state.engine

    @app.post("/run", status_code=202)
    def run(req: RunRequest, request: Request, background: BackgroundTasks) -> RunAccepted:
        e = eng(request)
        trace_id = uuid.uuid4().hex
        e.trace_dir.mkdir(parents=True, exist_ok=True)
        e.path(trace_id).touch()  # so an immediate GET /trace/{id} is "pending", not 404
        background.add_task(e.run, trace_id, req)  # sync fn: Starlette runs it in a threadpool
        return RunAccepted(trace_id=trace_id)

    @app.get("/trace/{trace_id}")
    def trace(trace_id: str, request: Request, after: int = -1) -> TraceView:
        """Events with seq > after, so a poller passes the last seq it saw."""
        e = eng(request)
        if not TRACE_ID.match(trace_id) or not e.path(trace_id).exists():
            raise HTTPException(404, "unknown trace")
        events = read_events(e.path(trace_id))
        return TraceView(
            trace_id=trace_id, events=[x for x in events if x.seq > after],
            done=any(x.type == "stop" for x in events),
        )

    @app.get("/replay")
    def replay(request: Request) -> list[str]:
        """Finished traces, newest first; replay mode fetches each via GET /trace/{id}."""
        e = eng(request)
        files = sorted(e.trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            p.stem for p in files
            if TRACE_ID.match(p.stem) and any(x.type == "stop" for x in read_events(p))
        ]

    @app.get("/health")
    def health(request: Request) -> dict[str, bool]:
        e = eng(request)
        llm = e.make_llm()
        try:
            llm_ok = llm.ping()
        finally:
            llm.close()
        try:
            qdrant_ok = e.retriever.client.collection_exists(COLLECTION)
        except Exception:  # noqa: BLE001 - any connection failure means "down"
            qdrant_ok = False
        return {"ok": llm_ok and qdrant_ok, "llm": llm_ok, "qdrant": qdrant_ok}

    return app


app = create_app()
