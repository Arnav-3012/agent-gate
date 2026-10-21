"use client";
import { CartesianGrid, ErrorBar, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { frontier, paretoPoints } from "@/lib/pareto";
import type { DomainResults } from "@/lib/results";

// X = cost per 1,000 queries, Y = F1 with bootstrap CI as error bars. Oracle router = the ceiling.
export function ParetoChart({ data }: { data: DomainResults }) {
  const pts = paretoPoints(data);
  const front = frontier(pts);
  const rows = pts.map((p) => ({ ...p, err: [p.f1 - p.lo, p.hi - p.f1] as [number, number], front: front.has(p.name) }));
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 24, left: 8 }}>
        <CartesianGrid stroke="var(--line)" />
        <XAxis type="number" dataKey="cost" name="cost / 1k" unit=" $" stroke="var(--muted)" label={{ value: "cost per 1,000 queries ($)", position: "bottom", fill: "var(--muted)" }} />
        <YAxis type="number" dataKey="f1" name="F1" domain={[0, 1]} stroke="var(--muted)" />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number) => v.toFixed(3)} labelFormatter={() => ""} />
        <Scatter name="arms" data={rows.filter((r) => r.kind === "arm")} fill="var(--accent)">
          <ErrorBar dataKey="err" direction="y" stroke="var(--accent)" />
        </Scatter>
        <Scatter name="oracle router" data={rows.filter((r) => r.kind === "oracle")} fill="var(--ok)" shape="diamond">
          <ErrorBar dataKey="err" direction="y" stroke="var(--ok)" />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
