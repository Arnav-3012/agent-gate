"use client";
import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";

export function Console() {
  const lines = useStore((s) => s.trace.lines);
  const answer = useStore((s) => s.trace.answer);
  const box = useRef<HTMLDivElement>(null);
  // scroll the log box only: scrollIntoView would drag the whole page down on mount
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines.length]);
  return (
    <section className="panel">
      <h2>Console</h2>
      <div className="console" ref={box}>
        {lines.length === 0 && <span className="empty">Ask a question to start.</span>}
        {lines.map((l) => <div key={l.seq}><span className="t">{l.t.toFixed(1)}s</span><span>{l.text}</span></div>)}
      </div>
      {answer && <p><b>Answer:</b> {answer}</p>}
    </section>
  );
}
