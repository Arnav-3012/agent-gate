"use client";
import { useEffect, useMemo, useState } from "react";
import { armLabel, isHosted } from "@/lib/arms";
import { useResults } from "@/lib/useResults";
import type { DomainResults } from "@/lib/results";
import { CountUp } from "../ui/CountUp";
import { Reveal } from "../ui/Reveal";
import { ParetoChart } from "./ParetoChart";

// Headline: the best fixed arm vs the oracle router (the ceiling a real router can chase).
export function BenchmarkView() {
  const { results, mock } = useResults();
  const [domain, setDomain] = useState("general");
  const [hosted, setHosted] = useState(true);
  useEffect(() => { if (!mock) setDomain((d) => (d in results ? d : Object.keys(results)[0])); }, [mock, results]);
  const full = results[domain] ?? Object.values(results)[0];
  const hasHosted = Object.keys(full.arms).some(isHosted);
  const d: DomainResults = useMemo(
    () => (hosted ? full : { ...full, arms: Object.fromEntries(Object.entries(full.arms).filter(([n]) => !isHosted(n))) }),
    [full, hosted],
  );
  const [best, bestS] = Object.entries(d.arms).sort((a, b) => b[1].f1 - a[1].f1)[0];
  const o = d.oracle_router;
  return (
    <>
      <section className="panel row" style={{ alignItems: "center", flexWrap: "wrap" }}>
        <select value={domain} onChange={(e) => setDomain(e.target.value)} aria-label="domain">
          {Object.keys(results).map((k) => <option key={k}>{k}</option>)}
        </select>
        {hasHosted && (
          <div className="seg" role="group" aria-label="arm set">
            <button className={!hosted ? "on" : ""} onClick={() => setHosted(false)}>Local arms</button>
            <button className={hosted ? "on" : ""} onClick={() => setHosted(true)}>+ Hosted</button>
          </div>
        )}
        <span className="empty">{mock ? "MOCK DATA: placeholder numbers, not results." : `${d.n} questions answered by every arm${hasHosted ? "" : " · hosted arms not run yet"}`}</span>
      </section>
      <section className="panel kpis">
        <div className="kpi"><b><CountUp value={bestS.f1} format={(v) => v.toFixed(3)} /></b><span>Best fixed F1 · {armLabel(best)}</span></div>
        {o && <div className="kpi"><b><CountUp value={o.f1} format={(v) => v.toFixed(3)} /></b><span>Oracle router F1</span></div>}
        {o && <div className="kpi"><b>+<CountUp value={o.gain_over_best_fixed} format={(v) => v.toFixed(3)} /></b><span>Headroom over best fixed</span></div>}
        {o && <div className="kpi"><b><CountUp value={o.cost_per_1k} format={(v) => `$${v.toFixed(3)}`} /></b><span>Oracle $/1k queries</span></div>}
        <div className="kpi"><b><CountUp value={bestS.cost_per_1k} format={(v) => `$${v.toFixed(3)}`} /></b><span>Best fixed $/1k</span></div>
      </section>
      <Reveal><section className="panel"><h2>Cost vs quality</h2><ParetoChart data={d} /><p className="empty">Filled dots are on the frontier (nothing is cheaper and better). Bars are 95% bootstrap intervals.</p></section></Reveal>
      <Reveal delay={80}>
        <section className="panel">
          <h2>Per arm</h2>
          <table>
            <thead><tr><th>arm</th><th>F1</th><th>EM</th><th>$/1k</th><th>p50 s</th><th>p95 s</th></tr></thead>
            <tbody>
              {Object.entries(d.arms).map(([n, s]) => (
                <tr key={n} className={n === best ? "best" : ""}><td>{armLabel(n)}</td><td>{s.f1.toFixed(3)}</td><td>{s.em.toFixed(3)}</td><td>{s.cost_per_1k.toFixed(3)}</td><td>{s.latency_p50.toFixed(1)}</td><td>{s.latency_p95.toFixed(1)}</td></tr>
              ))}
              {o && <tr className="oracle"><td>oracle router</td><td>{o.f1.toFixed(3)}</td><td>–</td><td>{o.cost_per_1k.toFixed(3)}</td><td>–</td><td>–</td></tr>}
            </tbody>
          </table>
        </section>
      </Reveal>
    </>
  );
}
