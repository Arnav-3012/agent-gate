"use client";
import { useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Line, OrbitControls, Sparkles, Text } from "@react-three/drei";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { ARM_META } from "@/lib/arms";
import { useStore } from "@/lib/store";

// Routing tunnel: the query sits in the centre, one node per arm on a ring (size = log price, colour = arm).
// A pulse travels the chosen path; nodes off the path dim once a route exists. Idle: slow drift + sparkles.
const R = 2.4;
type V3 = [number, number, number];
const pos = (i: number): V3 => {
  const a = (i / ARM_META.length) * Math.PI * 2 + Math.PI / 2;
  return [Math.cos(a) * R, Math.sin(a) * R * 0.8, 0];
};
const size = (usd: number) => 0.2 + 0.1 * Math.log10(usd / ARM_META[0].usdPer1k + 1);

function Pulse({ stops }: { stops: V3[] }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    const m = ref.current;
    if (!m || stops.length < 2) return void (m && (m.visible = false));
    const segs = stops.length - 1, p = (clock.elapsedTime * 0.6) % segs, i = Math.floor(p);
    m.visible = true;
    m.position.set(...(stops[i].map((v, k) => v + (stops[i + 1][k] - v) * (p - i)) as V3));
  });
  return <mesh ref={ref}><sphereGeometry args={[0.09, 16, 16]} /><meshBasicMaterial color="#ffffff" /></mesh>;
}

function Ring({ children }: { children: React.ReactNode }) {
  const g = useRef<THREE.Group>(null);
  useFrame((_, dt) => { if (g.current) g.current.rotation.z += dt * 0.03; });
  return <group ref={g}>{children}</group>;
}

export function Nebula() {
  const path = useStore((s) => s.trace.path);
  const done = useStore((s) => s.trace.done);
  const idx = (arm: string) => ARM_META.findIndex((a) => a.id === arm);
  const stops: V3[] = [[0, 0, 0], ...path.filter((a) => idx(a) >= 0).map((a) => pos(idx(a)))];
  const routed = path.length > 0;
  return (
    <div style={{ height: 380 }} className="canvas-wrap">
      <Canvas camera={{ position: [0, 0, 7.5], fov: 50 }} dpr={[1, 2]}>
        <color attach="background" args={["#08090a"]} />
        <fog attach="fog" args={["#08090a", 8, 16]} />
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 0, 4]} intensity={30} color="#7170ff" />
        <Sparkles count={80} scale={[10, 7, 4]} size={2} speed={0.3} opacity={0.5} color="#7170ff" />
        <Float speed={1.5} floatIntensity={0.4}>
          <mesh><sphereGeometry args={[0.28, 32, 32]} /><meshStandardMaterial color="#e8ecf1" emissive="#7170ff" emissiveIntensity={routed ? 1.2 : 0.5} /></mesh>
          <mesh scale={1.7}><sphereGeometry args={[0.28, 24, 24]} /><meshBasicMaterial color="#7170ff" transparent opacity={0.08} depthWrite={false} /></mesh>
        </Float>
        <Ring>
          {ARM_META.map((a, i) => {
            const hit = path.includes(a.id), last = hit && path[path.length - 1] === a.id;
            const s = size(a.usdPer1k) * (hit ? 1.35 : 1);
            return (
              <group key={a.id} position={pos(i)}>
                <mesh scale={s / 0.3}>
                  <sphereGeometry args={[0.3, 32, 32]} />
                  <meshStandardMaterial color={a.color} emissive={a.color} emissiveIntensity={hit ? 1.4 : routed ? 0.05 : 0.35} transparent opacity={routed && !hit ? 0.3 : 1} />
                </mesh>
                {hit && <mesh scale={s / 0.3 * 2}><sphereGeometry args={[0.3, 24, 24]} /><meshBasicMaterial color={a.color} transparent opacity={done && last ? 0.22 : 0.12} depthWrite={false} /></mesh>}
                <Text position={[0, -s - 0.28, 0]} fontSize={0.2} color={hit ? "#ffffff" : "#8b95a3"} anchorY="top">{a.label}</Text>
              </group>
            );
          })}
        </Ring>
        {stops.slice(1).map((p, i) => <Line key={i} points={[stops[i], p]} color="#7170ff" lineWidth={2} transparent opacity={0.8} />)}
        <Pulse stops={stops} />
        <OrbitControls enableZoom={false} enablePan={false} autoRotate={!routed} autoRotateSpeed={0.4} maxPolarAngle={Math.PI * 0.65} minPolarAngle={Math.PI * 0.35} />
        <EffectComposer>
          <Bloom intensity={routed ? 1.1 : 0.7} luminanceThreshold={0.35} luminanceSmoothing={0.4} mipmapBlur />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
