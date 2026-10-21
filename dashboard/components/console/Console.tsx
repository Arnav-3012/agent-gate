"use client";
import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";

export function Console() {
  const lines = useStore((s) => s.trace.lines);
  const answer = useStore((s) => s.trace.answer);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ block: "nearest" }), [lines.length]);
  return (
    <section className="panel">
      <h2>Console</h2>
      <div className="console">
        {lines.length === 0 && <span className="empty">Ask a question to start.</span>}
        {lines.map((l) => <div key={l.seq}><span className="t">{l.t.toFixed(1)}s</span><span>{l.text}</span></div>)}
        <div ref={end} />
      </div>
      {answer && <p><b>Answer:</b> {answer}</p>}
    </section>
  );
}
