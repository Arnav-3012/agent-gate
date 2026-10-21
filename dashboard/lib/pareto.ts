import type { DomainResults } from "./results";

export interface Point { name: string; cost: number; f1: number; lo: number; hi: number; kind: "arm" | "oracle" }

// Points on the cost/F1 plane; fixed arms plus the oracle-router ceiling.
export function paretoPoints(d: DomainResults): Point[] {
  const pts: Point[] = Object.entries(d.arms).map(([name, s]) => ({
    name, cost: s.cost_per_1k, f1: s.f1, lo: s.f1_ci[0], hi: s.f1_ci[1], kind: "arm",
  }));
  if (d.oracle_router) {
    const o = d.oracle_router;
    pts.push({ name: "oracle router", cost: o.cost_per_1k, f1: o.f1, lo: o.f1_ci[0], hi: o.f1_ci[1], kind: "oracle" });
  }
  return pts;
}

// Names of arms no other arm beats on both cost (lower) and F1 (higher).
export function frontier(pts: Point[]): Set<string> {
  const arms = pts.filter((p) => p.kind === "arm");
  return new Set(
    arms.filter((p) => !arms.some((q) => q !== p && q.cost <= p.cost && q.f1 >= p.f1 && (q.cost < p.cost || q.f1 > p.f1))).map((p) => p.name),
  );
}
