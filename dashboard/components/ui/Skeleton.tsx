// Shimmer placeholder while a 3D canvas / chart chunk loads (no layout jump: fixed height).
export function Skeleton({ height = 340, label = "loading" }: { height?: number; label?: string }) {
  return <div className="skeleton" style={{ height }} role="status" aria-label={label}><span>{label}…</span></div>;
}
