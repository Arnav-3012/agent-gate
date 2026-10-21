// Mock trace for development without an engine (shape = a real agent trace, schema v3).
import type { TraceEvent, LLMStats } from "./schema";

const llm = (p: number, c: number, cost: number): LLMStats => ({
  prompt_tokens: p, completion_tokens: c, cached_tokens: 0, latency_s: 0.8, cost_usd: cost,
});

export const MOCK_TRACE_ID = "0".repeat(32);
const t0 = 1_800_000_000;
const base = (seq: number, dt: number) => ({ trace_id: MOCK_TRACE_ID, seq, ts: t0 + dt });

export const MOCK_TRACE: TraceEvent[] = [
  { ...base(0, 0), type: "route_decided", route: "qwen7b-agent", p_agent: null, features: { n_entities: 2 }, reason: "comparison cue" },
  { ...base(1, 0.1), type: "plan", step: 0, subqueries: ["Who directed Inception?", "Who directed Tenet?"], llm: llm(420, 40, 0.00004) },
  { ...base(2, 0.9), type: "retrieve", step: 0, query: "Who directed Inception?", hits: [{ chunk_id: "a1", title: "Inception", score: 0.81 }], latency_s: 0.05 },
  { ...base(3, 1.0), type: "reflect", step: 0, confidence: 0.4, new_evidence: 1, next_query: "Who directed Tenet?", verdict: "continue", llm: llm(600, 30, 0.00006) },
  { ...base(4, 1.8), type: "retrieve", step: 1, query: "Who directed Tenet?", hits: [{ chunk_id: "b2", title: "Tenet", score: 0.79 }], latency_s: 0.05 },
  { ...base(5, 2.9), type: "answer", text: "Both were directed by Christopher Nolan.", llm: llm(900, 12, 0.00009) },
  { ...base(6, 3.0), type: "stop", reason: "confidence", steps: 2, total_tokens: 2002, total_latency_s: 3.0, total_cost_usd: 0.00019, error: null },
];
