# docs/explain: learning layer

Updated every time a file is added. Written/updated by `/sensei` after each phase step passes its exit criteria.

- [glossary.md](glossary.md): cumulative A-Z terms (definition, repo path, why, analogy)
- [concept-map.md](concept-map.md): how router, retriever, agent loop, tracing, ledger, AWS, dashboard connect
- [interview-prep.md](interview-prep.md): likely questions, honest short answers, numbers to quote

- [project-explanation.md](project-explanation.md): the problem, the question, what we build, why, measured facts, risks (start here)

## Phase files
- [phase-01-mlx-server.md](phase-01-mlx-server.md): local MLX LLM server, OpenAI-compatible API, cold vs warm latency, prefix caching
- [phase-01-llm-client.md](phase-01-llm-client.md): metered `LLMClient` (LLMCall, Meter), usage-based token counting, testing with MockTransport
- [phase-01-ingest-retriever.md](phase-01-ingest-retriever.md): corpus + gold ids, deterministic chunk ids, bge query prefix, Qdrant cosine retrieval, recall@5 by question type
- [phase-02-trace-schema.md](phase-02-trace-schema.md): frozen 7-event trace contract, `seq` + terminal `stop`, Emitter/JSONL sink, TS mirror and its parity test
- [phase-02-single-pass-rag.md](phase-02-single-pass-rag.md): `run_rag` (retrieve step 0 -> answer -> stop), shortest-answer prompt, error-to-stop, mocked tests, smoke 637 tok / 2.1 s
- [phase-02-agent-loop.md](phase-02-agent-loop.md): `run_agent` (plan -> retrieve/reflect -> answer), code-owned stop rules and caps, JSON retry, smoke: same answer as RAG at 2.8x tokens / ~7x latency
- [phase-02-cost-ledger.md](phase-02-cost-ledger.md): why cost is modelled not measured, `Pricing`/`Ledger`, Emitter stamping, smoke: RAG $0.0000661 vs agent $0.0001977 (3.0x)
- [phase-02-fastapi-engine-api.md](phase-02-fastapi-engine-api.md): `POST /run` 202 + background, `GET /trace/{id}?after`, `/replay`, `/health`; live: rag 783 tok $0.0000795, agent 2451 tok $0.0002671; the repeated-query waste seen in the trace
- [phase-02-smoke-eval.md](phase-02-smoke-eval.md): 30-Q go/no-go (NO-GO: agent F1 0.387 vs RAG 0.487 on bridge, 5.1x cost), why retrieval is not the bottleneck, bootstrap CI + oracle router, the reframe to a multi-arm router (D24)
- [phase-02-finance-corpus.md](phase-02-finance-corpus.md): FinanceBench, whitespace-collapsed 900-char windows, gold by evidence span, scoring rules, the accepted numeric limitation
- [phase-02-eval-tooling.md](phase-02-eval-tooling.md): metrics, resumable `smoke_eval`, evidence oracle, Pareto + bootstrap CI + oracle router, offline LLM judge, `make eval-all`
- [phase-03-eval-runners.md](phase-03-eval-runners.md): paired bootstrap, held-out cascade CV and hosted judge (D46), `run_baselines`, `benchmark.py` (results.json, finance headline split D38), judge validation with Cohen's kappa, tail-latency/SLO in `pareto.py`
- [phase-04-router-machinery.md](phase-04-router-machinery.md): arm registry + trace schema v3, query features, router labels, cascade, semantic cache, LinUCB bandit + offline replay, calibrated logistic classifier; all built on synthetic data, thresholds unfitted
- [phase-03-retrieval-scope.md](phase-03-retrieval-scope.md): retrieval fix test (company+fiscal-year scope, recall@k, one config for all arms, T1 results 35% -> 73%)
- [phase-03-results.md](phase-03-results.md): Phase 3 + arm set B results: 7-arm judged table, agent dominated, noise null for the oracle, budget curves, the one real router win (retrieve or not), claims and non-claims (D50-D55)
- [phase-03-router-probe-set.md](phase-03-router-probe-set.md): hypothesis-first probe set (categories, expected arm, abstention scoring), `draw/run/report`, why probe rows stay out of the headline tables
- [phase-07-reliability-scripts.md](phase-07-reliability-scripts.md): replay recorder, `decide_mode` live/replay fallback, failure drill, warm-up, demo script and README skeleton
- [phase-05-aws-layer.md](phase-05-aws-layer.md): HTTP API + two Lambdas + DynamoDB trace table (TTL) + $5 budget, deployed 2026-10-22 (CloudFront pending account verification); `DynamoSink`, shared secret, /health route. Per-service detail: `../aws/`, story: `../aws/deploy-log.md`
- [phase-06-dashboard-pages.md](phase-06-dashboard-pages.md): what each dashboard page and section shows (header pill, Command Center, Benchmark)

- [../aws/README.md](../aws/README.md): per-service AWS docs (IAM/STS, Budgets, CDK/CloudFormation, Lambda, API Gateway, DynamoDB, S3/CloudFront, CloudWatch, tunnel)
