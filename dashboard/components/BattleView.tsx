"use client";
import { useEffect, useRef, useState } from "react";
import { ARMS } from "@/lib/arms";
import { client } from "@/lib/engine";
import { pollTrace } from "@/lib/api";
import { MOCK_TRACE } from "@/lib/mock";
import { play } from "@/lib/replay";
import { useStore } from "@/lib/store";
import { emptyState, reduce, type TraceState } from "@/lib/trace";
import type { TraceEvent } from "@/lib/schema";

// Same question to two arms at once; each side has its own reducer state, so they race independently.
function Side({ arm, state }: { arm: string; state: TraceState }) {
  return (
    <section className="panel">
      <h2>{arm}</h2>
      <div className="kpis">
        <div className="kpi"><b>${state.costUsd.toFixed(5)}</b><span>cost</span></div>
        <div className="kpi"><b>{state.tokens}</b><span>tokens</span></div>
        <div className="kpi"><b>{state.steps}</b><span>steps</span></div>
        <div className="kpi"><b>{state.done ? "done" : state.lines.length ? "running" : "–"}</b><span>status</span></div>
      </div>
      <p><b>Answer:</b> {state.answer ?? "…"}</p>
      <div className="console">{state.lines.map((l) => <div key={l.seq}><span className="t">{l.t.toFixed(1)}s</span><span>{l.text}</span></div>)}</div>
    </section>
  );
}

export function BattleView() {
  const mode = useStore((s) => s.mode);
  const [q, setQ] = useState("Were the directors of Inception and Tenet the same person?");
  const [arms, setArms] = useState<[string, string]>([ARMS[0], ARMS[3]]);
  const [state, setState] = useState<[TraceState, TraceState]>([emptyState(), emptyState()]);
  const cancels = useRef<(() => void)[]>([]);
  useEffect(() => () => cancels.current.forEach((c) => c()), []);

  const feed = (i: 0 | 1) => (events: TraceEvent[]) =>
    setState((s) => { const n: [TraceState, TraceState] = [s[0], s[1]]; n[i] = events.reduce(reduce, s[i]); return n; });

  async function fight() {
    cancels.current.forEach((c) => c());
    setState([emptyState(), emptyState()]);
    if (mode === "live") {
      await Promise.all(([0, 1] as const).map(async (i) => pollTrace(client, await client.run(q, arms[i]), feed(i))));
    } else {
      // replay: the recorded trace at two speeds stands in for a fast and a slow arm
      cancels.current = [play(MOCK_TRACE, (e) => feed(0)([e]), { speed: 4 }), play(MOCK_TRACE, (e) => feed(1)([e]), { speed: 1.5 })];
    }
  }

  return (
    <>
      <section className="panel row">
        <input type="text" value={q} onChange={(e) => setQ(e.target.value)} aria-label="question" />
        {[0, 1].map((i) => (
          <select key={i} value={arms[i]} onChange={(e) => setArms((a) => (i ? [a[0], e.target.value] : [e.target.value, a[1]]))} aria-label={`arm ${i + 1}`}>
            {ARMS.map((a) => <option key={a}>{a}</option>)}
          </select>
        ))}
        <button onClick={fight} disabled={mode === "unavailable"}>Run both</button>
      </section>
      <div className="grid2"><Side arm={arms[0]} state={state[0]} /><Side arm={arms[1]} state={state[1]} /></div>
    </>
  );
}
