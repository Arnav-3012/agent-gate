# docs/explain: learning layer

Updated every time a file is added. Written/updated by `/sensei` after each phase step passes its exit criteria.

- [glossary.md](glossary.md): cumulative A-Z terms (definition, repo path, why, analogy)
- [concept-map.md](concept-map.md): how router, retriever, agent loop, tracing, ledger, AWS, dashboard connect
- [interview-prep.md](interview-prep.md): likely questions, honest short answers, numbers to quote

## Phase files
- [phase-01-mlx-server.md](phase-01-mlx-server.md): local MLX LLM server, OpenAI-compatible API, cold vs warm latency, prefix caching
- [phase-01-llm-client.md](phase-01-llm-client.md): metered `LLMClient` (LLMCall, Meter), usage-based token counting, testing with MockTransport
- [phase-01-ingest-retriever.md](phase-01-ingest-retriever.md): corpus + gold ids, deterministic chunk ids, bge query prefix, Qdrant cosine retrieval, recall@5 by question type
