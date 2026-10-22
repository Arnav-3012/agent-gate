# AgentGate

Cost-aware query router plus the evaluation harness that decides **when routing pays**. Seven answering arms (closed-book 7B, local Qwen 7B/14B RAG and agent loops, hosted Claude Haiku 5.5 and Sonnet 5.5 RAG), every query traced with cost and latency, and every router claim tested against random routing at the same spend.

## Findings (judged scores, general n=300 HotpotQA/SQuAD, finance n=100 FinanceBench text Qs)
| Arm (general) | Judged score | $/1k (true marginal) | Mean latency |
|---|---|---|---|
| Closed-book 7B | 0.337 | $0.0002 | 0.2 s |
| 7B RAG (local) | 0.745 | $0.004 | 3.6 s |
| 7B agent | 0.702 | $0.011 | 9.6 s |
| 14B RAG | 0.763 | $0.010 | 8.6 s |
| Haiku 5.5 RAG | 0.783 | $0.19 | 1.2 s |
| Sonnet 5.5 RAG | 0.808 | $3.78 | 1.7 s |

- **Agent loops do not pay here:** 7B agent vs 7B RAG -0.043 (paired 95% CI excludes 0) at 2.7x latency.
- **Learned retrieve-or-not gate works:** beats random routing at equal spend by up to +0.086 (CI excludes 0); vs always-retrieve it trades -0.037 for -22% latency.
- **Routing between similar-quality tiers does not:** 7B -> 14B / Haiku / Sonnet routers equal random on six signals; oracle gain never exceeds a noise null.
- **Finance:** no arm separable at n=100 (all paired CIs span 0).

Scores: Claude Haiku 5.5 judge (correct / partial / incorrect = 1 / 0.5 / 0), validated against 30 hand labels (Cohen's kappa 0.84). Cost: hosted arms at list price; local arms at energy cost (latency x 40 W x $0.10/kWh, an assumption in `config/pricing.yaml`); an API-equivalent view is kept as a sensitivity check. Rules for judging the router were fixed before the hosted-arm results (decision D54).

## Architecture
See [docs/architecture.md](docs/architecture.md) (kept untracked) or the summary: dashboard -> API Gateway -> Lambda router -> self-hosted engine (FastAPI, Qdrant, MLX Qwen 7B/14B, optional hosted Claude arms); every step is a typed trace event polled by the dashboard.

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
Finance results cover the 100 text-answerable FinanceBench questions; the 50 statement-arithmetic questions are excluded because no arm solves them. Small, closed-world corpora (HotpotQA distractor-style, 621-chunk finance) and n=100-300, so most differences between strong arms are within noise. Local cost uses an assumed power draw. The judge is an LLM (one family) validated on 30 finance rows only.
