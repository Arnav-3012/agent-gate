// Corpus map: /data/galaxy.json from scripts/umap_export.py (UMAP of the stored embeddings).
export interface GalaxyPoint { id: string; title: string; source: string; x: number; y: number; z: number }
export interface Galaxy { points: GalaxyPoint[] }

// Deterministic placeholder cloud for development; the page labels it as mock.
export function mockGalaxy(n = 600): Galaxy {
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const centres = [[-0.5, -0.4, 0], [0.5, 0.3, 0.2], [0, 0.6, -0.5]];
  return { points: Array.from({ length: n }, (_, i) => {
    const c = centres[i % 3];
    return { id: `m${i}`, title: `passage ${i}`, source: i % 3 === 0 ? "hotpotqa" : i % 3 === 1 ? "squad" : "finance", x: c[0] + rnd() * 0.3, y: c[1] + rnd() * 0.3, z: c[2] + rnd() * 0.3 };
  }) };
}

// Consecutive retrieval hops as arcs: top-1 chunk of hop i -> top-1 chunk of hop i+1 (found points only).
export function hopArcs(g: Galaxy, hops: string[][]): [GalaxyPoint, GalaxyPoint][] {
  const byId = new Map(g.points.map((p) => [p.id, p]));
  const tops = hops.map((h) => byId.get(h[0])).filter((p): p is GalaxyPoint => !!p);
  return tops.slice(1).map((p, i) => [tops[i], p]);
}
