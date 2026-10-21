# AgentGate

Smart query router: sends each query to the cheapest answering arm (RAG-7B, RAG-14B, agent loop, cache) that is good enough, with full cost and latency observability. Not "RAG vs agent": the measured cost-quality curve decides.

> Status: engine, eval tooling and router components are built; fitted router and headline numbers are pending the closing eval. Placeholders marked `TBD` are filled from `eval/results.json`, never estimated.

## Headline (TBD)
| | F1 | $/1k queries | p95 latency |
|---|---|---|---|
| Best fixed arm | TBD | TBD | TBD |
| Cheapest arm | TBD | TBD | TBD |
| Router | TBD | TBD | TBD |
| Oracle router (ceiling) | TBD | TBD | TBD |

Cost is modelled from token counts at a verified reference price (`config/pricing.yaml`), not billed.

## Architecture
See [docs/architecture.md](docs/architecture.md) (kept untracked) or the summary: dashboard -> API Gateway -> Lambda router -> self-hosted engine (FastAPI, Qdrant, MLX Qwen 7B/14B); every step is a typed trace event polled by the dashboard.

## Run
```
make up && make llm && make llm14     # Qdrant + both model servers
make ingest ingest-finance            # corpora
uv run uvicorn engine.api.main:app    # engine on :8000
python scripts/warmup.py              # health + one query per arm
make test
```

## Layout
`engine/` core (rag, agent, router, tracing, cost, api) · `eval/` metrics, baselines, Pareto, judge, probe set · `router_model/` classifier trainer · `scripts/` ingest, replay recorder, drill, warm-up · `config/` arms and pricing · `dashboard/` UI (in progress).

## Limitations
Finance results cover the 100 text-answerable FinanceBench questions; the 50 statement-arithmetic questions are excluded because no arm solves them. Small corpora; n per category is small, so see the confidence intervals.
