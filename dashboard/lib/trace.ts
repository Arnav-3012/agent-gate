// Pure trace reducer: folds events into the view state every dashboard panel reads.
import type { TraceEvent, StopReason } from "./schema";

export interface ConsoleLine {
  seq: number;
  t: number; // seconds since the first event
  kind: TraceEvent["type"];
  text: string;
}

export interface TraceState {
  route: string | null;
  steps: number;
  tokens: number;
  costUsd: number;
  latencyS: number;
  answer: string | null;
  stopReason: StopReason | null;
  done: boolean;
  lastSeq: number;
  t0: number | null;
  hops: string[][]; // chunk ids retrieved at each retrieve event, in order
  path: string[]; // arm ids visited: route, then each escalation target
  lines: ConsoleLine[];
}

export const emptyState = (): TraceState => ({
  route: null, steps: 0, tokens: 0, costUsd: 0, latencyS: 0,
  answer: null, stopReason: null, done: false, lastSeq: -1, t0: null, hops: [], path: [], lines: [],
});

const describe = (e: TraceEvent): string => {
  switch (e.type) {
    case "route_decided": return `route -> ${e.route} (${e.reason})`;
    case "plan": return `plan: ${e.subqueries.join(" | ")}`;
    case "retrieve": return `retrieve "${e.query}" -> ${e.hits.length} hits`;
    case "reflect": return `reflect: confidence ${e.confidence.toFixed(2)}, ${e.verdict}`;
    case "escalate": return `escalate ${e.from_mode} -> ${e.to_mode} (${e.reason})`;
    case "answer": return `answer: ${e.text}`;
    case "stop": return `stop: ${e.reason}`;
  }
};

const llmOf = (e: TraceEvent) => ("llm" in e ? e.llm : null);

// Idempotent per seq: a poller that re-delivers an event does not double-count it.
export function reduce(state: TraceState, e: TraceEvent): TraceState {
  if (e.seq <= state.lastSeq) return state;
  const t0 = state.t0 ?? e.ts;
  const next: TraceState = {
    ...state,
    lastSeq: e.seq,
    t0,
    lines: [...state.lines, { seq: e.seq, t: e.ts - t0, kind: e.type, text: describe(e) }],
  };
  const llm = llmOf(e);
  if (llm) {
    next.tokens += llm.prompt_tokens + llm.completion_tokens;
    next.costUsd += llm.cost_usd ?? 0;
  }
  if (e.type === "route_decided") {
    next.route = e.route;
    next.path = [e.route];
  }
  if (e.type === "retrieve") next.hops = [...state.hops, e.hits.map((h) => h.chunk_id)];
  if (e.type === "escalate") next.path = [...state.path, e.to_mode];
  if (e.type === "answer") next.answer = e.text;
  if (e.type === "stop") {
    // the engine's totals are authoritative once the run ends
    Object.assign(next, {
      done: true, stopReason: e.reason, steps: e.steps, tokens: e.total_tokens,
      latencyS: e.total_latency_s, costUsd: e.total_cost_usd ?? next.costUsd,
    });
  }
  return next;
}

export const reduceAll = (events: TraceEvent[]): TraceState => events.reduce(reduce, emptyState());
