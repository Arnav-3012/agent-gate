"use client";
import { ARM_META, armMeta, resultArmId } from "@/lib/arms";
import { useResults } from "@/lib/useResults";
import { useStore } from "@/lib/store";

const money = (v: number) => (v < 0.1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`);

// The hero panel: what the router saw, every candidate arm priced on a log scale, which one won and why.
export function RoutingPanel() {
  const t = useStore((s) => s.trace);
  const { results, mock } = useResults();
  const measured = mock ? {} : Object.fromEntries(Object.entries((results.general ?? Object.values(results)[0]).arms).map(([n, a]) => [resultArmId(n), a.cost_per_1k]));
  const price = (id: string, fallback: number) => measured[id] ?? fallback;
  const lo = Math.min(...ARM_META.map((a) => price(a.id, a.usdPer1k))), hi = Math.max(...ARM_META.map((a) => price(a.id, a.usdPer1k)));
  const MAX = Math.log10(hi / lo) || 1;
  const reason = t.lines.find((l) => l.kind === "route_decided")?.text.replace(/^route -> \S+ /, "") ?? "";
  const chosen = t.path[t.path.length - 1];
  const always = ARM_META[ARM_META.length - 1];
  const picked = chosen ? armMeta(chosen) : undefined;
  return (
    <section className="panel routing">
      <h2>Router decision</h2>
      <ol className="flow">
        <li className={t.route ? "on" : ""}><b>Query</b><span>features</span></li>
        <li className={t.route ? "on" : ""}><b>Router</b><span>{t.route ? reason || "decided" : t.done ? "no route event (default pipeline)" : "waiting"}</span></li>
        <li className={t.steps || t.done ? "on" : ""}><b>Arm</b><span>{picked?.label ?? "–"}</span></li>
        <li className={t.done ? "on" : ""}><b>Answer</b><span>{t.done ? `$${t.costUsd.toFixed(5)}` : "–"}</span></li>
      </ol>
      <div className="arms">
        {ARM_META.map((a) => {
          const w = (Math.log10(price(a.id, a.usdPer1k) / lo) / MAX) * 100;
          const hit = t.path.includes(a.id);
          return (
            <div key={a.id} className={`arm ${hit ? "hit" : t.route ? "dim" : ""}`} style={{ ["--c" as string]: a.color }}>
              <span className="name">{a.label}<small>{a.tier}</small></span>
              <span className="bar"><i style={{ width: `${Math.max(4, w)}%` }} /></span>
              <span className="price">{money(price(a.id, a.usdPer1k))}<small>/1k{measured[a.id] === undefined ? "*" : ""}</small></span>
            </div>
          );
        })}
      </div>
      {!mock && <p className="empty">* modelled price (not measured yet)</p>}
      {picked && t.done && (
        <p className="saving">
          {picked.id === always.id ? "Strongest arm used." : `${Math.round((1 - price(picked.id, picked.usdPer1k) / price(always.id, always.usdPer1k)) * 100)}% cheaper than always using ${always.label}`}
        </p>
      )}
    </section>
  );
}
