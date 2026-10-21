"use client";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { mockGalaxy, type Galaxy as G } from "@/lib/galaxy";
import { MOCK_TRACE } from "@/lib/mock";
import { play } from "@/lib/replay";
import { useStore } from "@/lib/store";
import { Console } from "./console/Console";

const Galaxy = dynamic(() => import("./three/Galaxy").then((m) => m.Galaxy), { ssr: false });

export function GalaxyView() {
  const [data, setData] = useState<G>(() => mockGalaxy());
  const [mock, setMock] = useState(true);
  const { push, reset } = useStore();
  const cancel = useRef<() => void>(() => {});
  useEffect(() => {
    fetch("/data/galaxy.json").then((r) => (r.ok ? r.json() : Promise.reject())).then((g: G) => { setData(g); setMock(false); }).catch(() => {});
    return () => cancel.current();
  }, []);
  // The mock trace's chunk ids exist only in the mock cloud; real traces come from the Command Center.
  const demo = () => { cancel.current(); reset(); cancel.current = play(MOCK_TRACE.map((e) => (e.type === "retrieve" ? { ...e, hits: e.hits.map((h, i) => ({ ...h, chunk_id: `m${(e.step * 7 + i) * 11 + 3}` })) } : e)), (e) => push([e]), { speed: 2 }); };
  return (
    <>
      <section className="panel row" style={{ alignItems: "center" }}>
        <button onClick={demo}>Replay a query</button>
        <span className="empty">{mock ? "MOCK CLOUD: run `make umap` to map the real corpus." : `${data.points.length} passages`}</span>
      </section>
      <section className="panel"><h2>Corpus Galaxy</h2><Galaxy data={data} /></section>
      <Console />
    </>
  );
}
