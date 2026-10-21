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
