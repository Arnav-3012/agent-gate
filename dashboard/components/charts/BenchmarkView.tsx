"use client";
import { useEffect, useState } from "react";
import { MOCK_RESULTS, type Results } from "@/lib/results";
import { ParetoChart } from "./ParetoChart";

// Loads /data/results.json (copy of eval/results.json) and falls back to labelled mock numbers.
export function BenchmarkView() {
  const [results, setResults] = useState<Results>(MOCK_RESULTS);
  const [mock, setMock] = useState(true);
  const [domain, setDomain] = useState("general");
  useEffect(() => {
    fetch("/data/results.json").then((r) => (r.ok ? r.json() : Promise.reject())).then((d: Results) => {
      setResults(d); setMock(false); setDomain(Object.keys(d)[0]);
    }).catch(() => {});
  }, []);
  const d = results[domain];
  return (
    <>
      <section className="panel row" style={{ alignItems: "center" }}>
        <select value={domain} onChange={(e) => setDomain(e.target.value)} aria-label="domain">
          {Object.keys(results).map((k) => <option key={k}>{k}</option>)}
        </select>
        <span className="empty">{mock ? "MOCK DATA: placeholder numbers, not results." : `${d.n} questions answered by every arm`}</span>
      </section>
      <section className="panel"><h2>Cost vs quality</h2><ParetoChart data={d} /></section>
      <section className="panel">
        <h2>Per arm</h2>
        <table>
          <thead><tr><th>arm</th><th>F1</th><th>EM</th><th>$/1k</th><th>p50 s</th><th>p95 s</th></tr></thead>
          <tbody>
            {Object.entries(d.arms).map(([n, s]) => (
              <tr key={n}><td>{n}</td><td>{s.f1.toFixed(3)}</td><td>{s.em.toFixed(3)}</td><td>{s.cost_per_1k.toFixed(3)}</td><td>{s.latency_p50.toFixed(1)}</td><td>{s.latency_p95.toFixed(1)}</td></tr>
            ))}
            {d.oracle_router && (
              <tr><td>oracle router</td><td>{d.oracle_router.f1.toFixed(3)}</td><td>–</td><td>{d.oracle_router.cost_per_1k.toFixed(3)}</td><td>–</td><td>–</td></tr>
            )}
          </tbody>
        </table>
      </section>
    </>
  );
}
