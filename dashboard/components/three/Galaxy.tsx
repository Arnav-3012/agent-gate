"use client";
import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { hopArcs, type Galaxy as G } from "@/lib/galaxy";
import { useStore } from "@/lib/store";

const COLOR: Record<string, string> = { hotpotqa: "#6c8cff", squad: "#e3a93c", finance: "#3ecf8e" };

// Every passage is a dot (coloured by source); chunks retrieved by the current query glow white and
// consecutive hops are joined by arcs, so multi-hop retrieval is visible as a path across the map.
export function Galaxy({ data }: { data: G }) {
  const hops = useStore((s) => s.trace.hops);
  const hit = useMemo(() => new Set(hops.flat()), [hops]);
  const arcs = hopArcs(data, hops);
  const { pos, col } = useMemo(() => {
    const pos = new Float32Array(data.points.length * 3);
    const col = new Float32Array(data.points.length * 3);
    data.points.forEach((p, i) => {
      pos.set([p.x * 3, p.y * 3, p.z * 3], i * 3);
      const c = hit.has(p.id) ? [1, 1, 1] : hexToRgb(COLOR[p.source] ?? "#8b95a3").map((v) => v * 0.55);
      col.set(c, i * 3);
    });
    return { pos, col };
  }, [data, hit]);
  return (
    <div style={{ height: 520 }}>
      <Canvas camera={{ position: [0, 0, 7], fov: 50 }}>
        <points key={hit.size}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[pos, 3]} />
            <bufferAttribute attach="attributes-color" args={[col, 3]} />
          </bufferGeometry>
          <pointsMaterial vertexColors size={0.06} sizeAttenuation />
        </points>
        {arcs.map(([a, b], i) => (
          <Line key={i} points={[[a.x * 3, a.y * 3, a.z * 3], [b.x * 3, b.y * 3, b.z * 3]]} color="#ffffff" lineWidth={2} />
        ))}
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}

function hexToRgb(h: string): number[] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
