"use client";
import { Cell, CartesianGrid, ErrorBar, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { armLabel, armMeta, resultArmId } from "@/lib/arms";
import { frontier, paretoPoints } from "@/lib/pareto";
import type { DomainResults } from "@/lib/results";

// X = cost per 1,000 queries (log), Y = F1 with bootstrap CI error bars. Frontier arms are filled; oracle router = the ceiling.
export function ParetoChart({ data }: { data: DomainResults }) {
  const pts = paretoPoints(data);
  const front = frontier(pts);
  const rows = pts.map((p) => ({ ...p, label: p.kind === "oracle" ? "Oracle router" : armLabel(p.name), err: [p.f1 - p.lo, p.hi - p.f1] as [number, number], front: front.has(p.name), color: armMeta(resultArmId(p.name))?.color ?? "#7170ff" }));
  const log = rows.every((r) => r.cost > 0);
  const lo = Math.min(...rows.map((r) => r.cost)), hi = Math.max(...rows.map((r) => r.cost));
  const ticks = log ? [...new Set(rows.map((r) => +r.cost.toPrecision(2)))].sort((a, b) => a - b) : undefined;
  return (
    <ResponsiveContainer width="100%" height={380}>
      <ScatterChart margin={{ top: 16, right: 40, bottom: 28, left: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,.06)" />
        <XAxis type="number" dataKey="cost" name="cost / 1k" scale={log ? "log" : "auto"} domain={log ? [lo / 1.5, hi * 1.5] : [0, "auto"]} ticks={ticks} tickFormatter={(v: number) => `$${v.toFixed(2)}`} stroke="var(--muted)" label={{ value: "cost per 1,000 queries (log scale)", position: "bottom", fill: "var(--muted)", fontSize: 12 }} />
        <YAxis type="number" dataKey="f1" name="F1" domain={[0, 1]} stroke="var(--muted)" />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#191a1b", border: "1px solid rgba(255,255,255,.08)", borderRadius: 8 }} formatter={(v: number) => v.toFixed(3)} labelFormatter={() => ""} />
        <Scatter name="arms" shape="circle" data={rows.filter((r) => r.kind === "arm")} isAnimationActive animationDuration={900}>
          {rows.filter((r) => r.kind === "arm").map((r) => <Cell key={r.name} fill={r.front ? r.color : "transparent"} stroke={r.color} strokeWidth={2} />)}
          <ErrorBar dataKey="err" direction="y" stroke="rgba(255,255,255,.35)" />
          <LabelList dataKey="label" position="top" fill="#d0d6e0" stroke="none" fontSize={11} fontWeight={510} offset={12} />
        </Scatter>
        <Scatter name="oracle router" data={rows.filter((r) => r.kind === "oracle")} fill="#10b981" shape="diamond" isAnimationActive animationDuration={900}>
          <ErrorBar dataKey="err" direction="y" stroke="#10b981" />
          <LabelList dataKey="label" position="top" fill="#10b981" stroke="none" fontSize={11} fontWeight={510} offset={12} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
