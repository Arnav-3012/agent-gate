"use client";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { client } from "@/lib/engine";
import { pollTrace } from "@/lib/api";
import { MOCK_TRACE } from "@/lib/mock";
import { play } from "@/lib/replay";
import { useStore } from "@/lib/store";
import { Console } from "./console/Console";
import { LiveRun } from "./console/LiveRun";
import { Timeline } from "./console/Timeline";
import { Reveal } from "./ui/Reveal";
import { KpiStrip } from "./ui/KpiStrip";
import { RoutingPanel } from "./ui/RoutingPanel";
import { Skeleton } from "./ui/Skeleton";

const Nebula = dynamic(() => import("./three/Nebula").then((m) => m.Nebula), { ssr: false, loading: () => <Skeleton label="loading routing view" /> });

export function CommandCenter() {
  const [q, setQ] = useState("Were the directors of Inception and Tenet the same person?");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const { mode, push, reset } = useStore();
  const cancel = useRef<() => void>(() => {});
  useEffect(() => () => cancel.current(), []);

  const replay = () => {
    cancel.current = play(MOCK_TRACE, (e) => push([e]), { speed: 2, onDone: () => setBusy(false) });
  };

  async function ask() {
    cancel.current();
    reset();
    setNotice("");
    setBusy(true);
    if (mode !== "live") return replay();
    try {
      // a run that ends without a stop event (poll gave up) counts as a failure too
      if (!(await pollTrace(client, await client.run(q), push))) throw new Error("timed out");
      setBusy(false);
    } catch {
      // engine died or timed out mid-run: same reducer, recorded events, so the demo keeps going
      reset();
      setNotice("Engine failed mid-run: replaying a recorded trace instead.");
      replay();
    }
  }

  return (
    <>
      <section className="hero">
        <span className="kicker">Smart query router</span>
        <h1>Pay for the cheapest model<br />that is good enough.</h1>
        <p>AgentGate picks, per question, among seven arms from a 7B local model to hosted Sonnet, and shows every decision as it happens.</p>
        <div className="ask">
          <input type="text" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !busy && ask()} aria-label="question" />
          <button onClick={ask} disabled={busy || mode === "unavailable"}>{busy ? "Running…" : "Ask"}</button>
        </div>
        {notice && <p className="notice" role="status">{notice}</p>}
      </section>
      <KpiStrip />
      <div className="grid2">
        <Reveal><div className="stack"><section className="panel"><h2>Routing nebula</h2><Nebula /></section><RoutingPanel /></div></Reveal>
        <Reveal delay={80}><div className="stack"><LiveRun /><Timeline /></div></Reveal>
      </div>
      <Reveal><Console /></Reveal>
      <section className="how">
        <Reveal><h2 className="big">How a query is routed</h2></Reveal>
        {[
          ["01", "Featurise", "Cheap signals from the question (entities, comparison cues, length) feed the router. No LLM call yet."],
          ["02", "Choose an arm", "A classifier trained on measured F1 per arm picks the cheapest arm expected to be good enough."],
          ["03", "Escalate only if needed", "If the arm is unsure after reflecting, the query moves up the ladder instead of starting expensive."],
          ["04", "Trace everything", "Each step is an event: cost, tokens, latency. This page is just those events, replayed live."],
        ].map(([n, t, d], i) => (
          <Reveal key={n} delay={i * 70}><article className="card"><span className="num">{n}</span><h3>{t}</h3><p>{d}</p></article></Reveal>
        ))}
      </section>
    </>
  );
}
