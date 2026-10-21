// Frozen trace contract. Mirror of engine/tracing/schema.py. Change both + SCHEMA_VERSION together.
export const SCHEMA_VERSION = 3;

export type Mode = "rag" | "agent";
export type StopReason =
  | "rag_done"
  | "confidence"
  | "no_new_evidence"
  | "max_steps"
  | "max_tokens"
  | "error";

export interface LLMStats {
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  latency_s: number;
  cost_usd: number | null;
}

export interface Hit {
  chunk_id: string;
  title: string;
  score: number;
}

interface Base {
  trace_id: string;
  seq: number;
  ts: number;
}

export interface RouteDecided extends Base {
  type: "route_decided";
  route: string; // arm id from config/arms.yaml (v2: "rag" | "agent")
  p_agent: number | null;
  features: Record<string, number>;
  reason: string;
}

export interface Plan extends Base {
  type: "plan";
  step: number;
  subqueries: string[];
  llm: LLMStats;
}

export interface Retrieve extends Base {
  type: "retrieve";
  step: number;
  query: string;
  hits: Hit[];
  latency_s: number;
}

export interface Reflect extends Base {
  type: "reflect";
  step: number;
  confidence: number;
  new_evidence: number;
  next_query: string; // "" = nothing; v2
  verdict: "continue" | "stop";
  llm: LLMStats;
}

export interface Escalate extends Base {
  type: "escalate";
  from_mode: string; // arm ids since v3
  to_mode: string;
  reason: string;
}

export interface Answer extends Base {
  type: "answer";
  text: string;
  llm: LLMStats;
}

export interface Stop extends Base {
  type: "stop";
  reason: StopReason;
  steps: number;
  total_tokens: number;
  total_latency_s: number;
  total_cost_usd: number | null;
  error: string | null;
}

export type TraceEvent = RouteDecided | Plan | Retrieve | Reflect | Escalate | Answer | Stop;

export const isTerminal = (e: TraceEvent): e is Stop => e.type === "stop";
