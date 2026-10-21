"use client";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { client } from "@/lib/engine";
import { pollTrace } from "@/lib/api";
import { MOCK_TRACE } from "@/lib/mock";
import { play } from "@/lib/replay";
import { useStore } from "@/lib/store";
import { Console } from "./console/Console";
import { Timeline } from "./console/Timeline";
import { KpiStrip } from "./ui/KpiStrip";

const Nebula = dynamic(() => import("./three/Nebula").then((m) => m.Nebula), { ssr: false });

export function CommandCenter() {
  const [q, setQ] = useState("Were the directors of Inception and Tenet the same person?");
  const [busy, setBusy] = useState(false);
  const { mode, push, reset } = useStore();
  const cancel = useRef<() => void>(() => {});
  useEffect(() => () => cancel.current(), []);

  async function ask() {
    cancel.current();
    reset();
    setBusy(true);
    try {
      if (mode === "live") {
        await pollTrace(client, await client.run(q), push);
        setBusy(false);
      } else {
        // replay fallback: same reducer, recorded events
        cancel.current = play(MOCK_TRACE, (e) => push([e]), { speed: 2, onDone: () => setBusy(false) });
      }
    } catch {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="panel">
        <div className="row">
          <input type="text" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !busy && ask()} aria-label="question" />
          <button onClick={ask} disabled={busy || mode === "unavailable"}>{busy ? "Running…" : "Ask"}</button>
        </div>
      </section>
      <KpiStrip />
      <div className="grid2">
        <section className="panel"><h2>Routing Nebula</h2><Nebula /></section>
        <div style={{ display: "grid", gap: 16, alignContent: "start" }}><Timeline /><Console /></div>
      </div>
    </>
  );
}
