"use client";
import { useStore } from "@/lib/store";

const COLOR: Record<string, string> = {
  route_decided: "var(--accent)", plan: "var(--muted)", retrieve: "var(--ok)", reflect: "var(--warn)",
  escalate: "var(--bad)", answer: "var(--accent)", stop: "var(--ink)",
};

// Each event is a tick at its time offset; width of the track = the run so far.
export function Timeline() {
  const lines = useStore((s) => s.trace.lines);
  const span = Math.max(1, ...lines.map((l) => l.t));
  return (
    <section className="panel">
      <h2>Timeline</h2>
      <svg viewBox="0 0 100 14" width="100%" height="48" role="img" aria-label="event timeline">
        <line x1="0" y1="7" x2="100" y2="7" stroke="var(--line)" />
        {lines.map((l) => (
          <circle key={l.seq} cx={(l.t / span) * 98 + 1} cy="7" r="1.6" fill={COLOR[l.kind]}><title>{`${l.t.toFixed(1)}s ${l.kind}`}</title></circle>
        ))}
      </svg>
      <div className="row" style={{ flexWrap: "wrap", fontSize: 12, color: "var(--muted)" }}>
        {Object.keys(COLOR).map((k) => <span key={k}><span style={{ color: COLOR[k] }}>●</span> {k}</span>)}
      </div>
    </section>
  );
}
