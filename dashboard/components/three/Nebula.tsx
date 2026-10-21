"use client";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import { ARMS } from "@/lib/arms";
import { useStore } from "@/lib/store";

// Query node in the centre, one node per arm on a ring. The chosen arm lights up and a line forks to it;
// each escalation adds a line from the previous arm to the next.
const R = 2.2;
const pos = (i: number): [number, number, number] => {
  const a = (i / ARMS.length) * Math.PI * 2 + Math.PI / 4;
  return [Math.cos(a) * R, Math.sin(a) * R, 0];
};

export function Nebula() {
  const path = useStore((s) => s.trace.path);
  const done = useStore((s) => s.trace.done);
  const at = (arm: string): [number, number, number] => pos(ARMS.indexOf(arm as (typeof ARMS)[number]));
  const stops = [[0, 0, 0] as [number, number, number], ...path.filter((a) => ARMS.includes(a as never)).map(at)];
  return (
    <div style={{ height: 340 }}>
      <Canvas camera={{ position: [0, 0, 7], fov: 50 }}>
        <ambientLight intensity={0.8} />
        <pointLight position={[5, 5, 5]} />
        <mesh><sphereGeometry args={[0.25, 24, 24]} /><meshStandardMaterial color="#8b95a3" /></mesh>
        {ARMS.map((arm, i) => {
          const hit = path.includes(arm);
          return (
            <group key={arm} position={pos(i)}>
              <mesh scale={hit ? 1.4 : 1}>
                <sphereGeometry args={[0.3, 24, 24]} />
                <meshStandardMaterial color={hit ? (done && path[path.length - 1] === arm ? "#3ecf8e" : "#6c8cff") : "#3a4252"} emissive={hit ? "#6c8cff" : "#000"} emissiveIntensity={hit ? 0.6 : 0} />
              </mesh>
              <Text position={[0, -0.6, 0]} fontSize={0.22} color="#8b95a3">{arm}</Text>
            </group>
          );
        })}
        {stops.slice(1).map((p, i) => <Line key={i} points={[stops[i], p]} color="#6c8cff" lineWidth={2} />)}
        <OrbitControls enableZoom={false} enablePan={false} />
      </Canvas>
    </div>
  );
}
