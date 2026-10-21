import { describe, expect, it } from "vitest";
import { MOCK_TRACE } from "./mock";
import { reduceAll, reduce, emptyState } from "./trace";
import { gaps, play, type Scheduler } from "./replay";
import { pollTrace } from "./api";

describe("reduce", () => {
  it("folds a full trace into totals from the stop event", () => {
    const s = reduceAll(MOCK_TRACE);
    expect(s.done).toBe(true);
    expect(s.route).toBe("qwen7b-agent");
    expect(s.answer).toBe("Both were directed by Christopher Nolan.");
    expect(s.tokens).toBe(2002);
    expect(s.lines).toHaveLength(MOCK_TRACE.length);
  });
  it("ignores a re-delivered event", () => {
    const once = reduce(emptyState(), MOCK_TRACE[0]);
    expect(reduce(once, MOCK_TRACE[0])).toBe(once);
  });
  it("accumulates tokens and cost before stop", () => {
    const s = reduceAll(MOCK_TRACE.slice(0, 4));
    expect(s.tokens).toBe(460 + 630);
    expect(s.costUsd).toBeCloseTo(0.0001);
    expect(s.done).toBe(false);
  });
});

describe("replay", () => {
  it("caps gaps and scales by speed", () => {
    const g = gaps(MOCK_TRACE, 1, 1500);
    [0, 100, 800, 100, 800, 1100, 100].forEach((want, i) => expect(g[i]).toBeCloseTo(want, 3));
    expect(gaps(MOCK_TRACE, 2)[2]).toBeCloseTo(400, 3);
    expect(gaps(MOCK_TRACE, 1, 500)[5]).toBe(500);
  });
  it("delivers events in order and can be cancelled", () => {
    const queue: (() => void)[] = [];
    const sched: Scheduler = { setTimeout: (fn) => (queue.push(fn), queue.length), clearTimeout: () => {} };
    const seen: number[] = [];
    const cancel = play(MOCK_TRACE, (e) => seen.push(e.seq), { scheduler: sched });
    while (queue.length && seen.length < 3) queue.shift()!();
    expect(seen).toEqual([0, 1, 2]);
    cancel();
    queue.shift()?.();
    expect(seen).toEqual([0, 1, 2]);
  });
});

describe("pollTrace", () => {
  it("passes the last seen seq as `after` until done", async () => {
    const afters: number[] = [];
    const client = {
      trace: async (_id: string, after = -1) => {
        afters.push(after);
        const events = MOCK_TRACE.filter((e) => e.seq > after).slice(0, 3);
        return { trace_id: "x", events, done: events.some((e) => e.type === "stop") };
      },
    };
    const got: number[] = [];
    const ok = await pollTrace(client, "x", (es) => got.push(...es.map((e) => e.seq)), { sleep: async () => {} });
    expect(ok).toBe(true);
    expect(afters).toEqual([-1, 2, 5]);
    expect(got).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });
});

import { MOCK_RESULTS } from "./results";
import { frontier, paretoPoints } from "./pareto";

describe("pareto", () => {
  it("marks dominated arms", () => {
    const pts = paretoPoints({ ...MOCK_RESULTS.general, arms: {
      ...MOCK_RESULTS.general.arms,
      bad: { ...MOCK_RESULTS.general.arms["qwen7b-rag"], cost_per_1k: 1, f1: 0.3 },
    } });
    const f = frontier(pts);
    expect(f.has("bad")).toBe(false);
    expect(f.has("qwen7b-rag")).toBe(true);
    expect(pts.some((p) => p.kind === "oracle")).toBe(true);
  });
});
describe("path", () => {
  it("tracks escalations", () => {
    const s = reduceAll([
      { trace_id: "x", seq: 0, ts: 1, type: "route_decided", route: "qwen7b-rag", p_agent: null, features: {}, reason: "r" },
      { trace_id: "x", seq: 1, ts: 2, type: "escalate", from_mode: "qwen7b-rag", to_mode: "qwen14b-rag", reason: "weak" },
    ]);
    expect(s.path).toEqual(["qwen7b-rag", "qwen14b-rag"]);
    expect(s.lines[1].t).toBe(1);
  });
});

import { hopArcs, mockGalaxy } from "./galaxy";
describe("galaxy", () => {
  it("records hop chunk ids and joins top hits with arcs", () => {
    const s = reduceAll(MOCK_TRACE);
    expect(s.hops).toEqual([["a1"], ["b2"]]);
    const g = { points: [{ id: "a1", title: "", source: "s", x: 0, y: 0, z: 0 }, { id: "b2", title: "", source: "s", x: 1, y: 1, z: 1 }] };
    expect(hopArcs(g, s.hops)).toHaveLength(1);
    expect(mockGalaxy(30).points).toHaveLength(30);
  });
});
