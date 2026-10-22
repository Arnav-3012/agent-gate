"use client";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { armMeta } from "@/lib/arms";
import type { TraceEvent } from "@/lib/schema";
import { useStore } from "@/lib/store";

const STAGES = ["route_decided", "plan", "retrieve", "reflect", "escalate", "answer", "stop"] as const;
const LABEL: Record<TraceEvent["type"], string> = {
  route_decided: "Route", plan: "Plan", retrieve: "Retrieve", reflect: "Reflect", escalate: "Escalate", answer: "Answer", stop: "Stop",
};

function Typewriter({ text }: { text: string }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
    const id = setInterval(() => setN((v) => (v >= text.length ? (clearInterval(id), v) : v + 2)), 18);
    return () => clearInterval(id);
  }, [text]);
  return <>{text.slice(0, n)}<i className="caret" /></>;
}

function Body({ e }: { e: TraceEvent }) {
  switch (e.type) {
    case "route_decided": {
      const a = armMeta(e.route);
      return (
        <>
          <div className="chips">
            <span className="chip" style={{ ["--c" as string]: a?.color }}>{a?.label ?? e.route}</span>
            {e.p_agent != null && <span className="chip dim">p(agent) {e.p_agent.toFixed(2)}</span>}
            {Object.entries(e.features).slice(0, 4).map(([k, v]) => <span key={k} className="chip dim">{k} {v}</span>)}
          </div>
          <p className="why">{e.reason}</p>
        </>
      );
    }
    case "plan": return <ul className="subq">{e.subqueries.map((q) => <li key={q}>{q}</li>)}</ul>;
    case "retrieve":
      return (
        <>
          <p className="why">“{e.query}” · {(e.latency_s * 1000).toFixed(0)} ms</p>
          {e.hits.map((h) => (
            <div className="hit" key={h.chunk_id}>
              <span>{h.title}</span>
              <span className="bar"><i style={{ width: `${Math.min(100, h.score * 100)}%` }} /></span>
              <span className="mono">{h.score.toFixed(2)}</span>
            </div>
          ))}
        </>
      );
    case "reflect":
      return (
        <div className="hit">
          <span>confidence</span>
          <span className="bar"><i style={{ width: `${e.confidence * 100}%`, background: e.verdict === "stop" ? "var(--ok)" : "var(--warn)" }} /></span>
          <span className="mono">{e.confidence.toFixed(2)} · {e.verdict}</span>
        </div>
      );
    case "escalate": return <p className="why">{e.from_mode} → <b>{e.to_mode}</b> · {e.reason}</p>;
    case "answer": return <p className="answer"><Typewriter text={e.text} /></p>;
    case "stop": return <p className="why">{e.reason} · {e.steps} steps · {e.total_tokens} tokens · {e.total_latency_s.toFixed(1)} s</p>;
  }
}

// The live artefact: one card per trace event, appearing as the engine emits it, plus a stage rail on top.
export function LiveRun() {
  const events = useStore((s) => s.trace.events);
  const lines = useStore((s) => s.trace.lines);
  const box = useRef<HTMLOListElement>(null);
  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);
  const done = useStore((s) => s.trace.done);
  const running = events.length > 0 && !done;
  const seen = new Set(events.map((e) => e.type));
  const reached = STAGES.filter((s) => seen.has(s)).length;
  return (
    <section className="panel liverun">
      <h2>Live run <span className={`livedot ${running ? "run" : done ? "done" : ""}`}>{running ? "streaming" : done ? "complete" : "idle"}</span></h2>
      <div className="stepper" style={{ ["--p" as string]: `${(Math.max(0, reached - 1) / (STAGES.length - 1)) * 100}%` }}>
        <i className="track"><i className="fillbar" /></i>
        <ol className="rail">
          {STAGES.map((s) => <li key={s} className={seen.has(s) ? "on" : ""}><span className="node" />{LABEL[s]}</li>)}
        </ol>
      </div>
      {events.length === 0 ? <p className="empty">Ask a question: each router step appears here as it happens.</p> : (
        <ol className="steps" ref={box}>
          <AnimatePresence initial={false}>
          {events.map((e, i) => (
            <motion.li key={e.seq} layout className={`step ${e.type} ${i === events.length - 1 && running ? "latest" : ""}`} initial={{ opacity: 0, x: -14, scale: 0.97 }} animate={{ opacity: 1, x: 0, scale: 1 }} transition={{ type: "spring", stiffness: 380, damping: 32 }}>
              <div className="head"><b>{LABEL[e.type]}</b>{"step" in e && <span className="chip dim">step {e.step}</span>}<span className="mono t">{lines[i]?.t.toFixed(1)}s</span></div>
              <Body e={e} />
            </motion.li>
          ))}
          </AnimatePresence>
          {running && <li className="thinking" aria-label="waiting for next event"><span /><span /><span /></li>}
        </ol>
      )}
    </section>
  );
}
