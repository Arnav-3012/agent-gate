"use client";
import { useStore } from "@/lib/store";

export function KpiStrip() {
  const t = useStore((s) => s.trace);
  const items: [string, string][] = [
    ["Arm", t.path.join(" → ") || "–"],
    ["Cost", `$${t.costUsd.toFixed(5)}`],
    ["Tokens", String(t.tokens)],
    ["Latency", `${t.latencyS.toFixed(1)} s`],
    ["Steps", String(t.steps)],
  ];
  return (
    <section className="panel kpis">
      {items.map(([k, v]) => <div className="kpi" key={k}><b>{v}</b><span>{k}</span></div>)}
    </section>
  );
}
