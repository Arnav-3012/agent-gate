"use client";
import { useStore } from "@/lib/store";
import { CountUp } from "./CountUp";

export function KpiStrip() {
  const t = useStore((s) => s.trace);
  return (
    <section className="panel kpis">
      <div className="kpi"><b className="arm-path">{t.path.join(" → ") || "–"}</b><span>Arm path</span></div>
      <div className="kpi"><b><CountUp value={t.costUsd} format={(v) => `$${v.toFixed(5)}`} /></b><span>Cost</span></div>
      <div className="kpi"><b><CountUp value={t.tokens} format={(v) => Math.round(v).toString()} /></b><span>Tokens</span></div>
      <div className="kpi"><b><CountUp value={t.latencyS} format={(v) => `${v.toFixed(1)} s`} /></b><span>Latency</span></div>
      <div className="kpi"><b>{t.steps}</b><span>Steps</span></div>
    </section>
  );
}
