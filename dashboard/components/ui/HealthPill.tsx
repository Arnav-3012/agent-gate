"use client";
import { useStore } from "@/lib/store";
import { useModePolling } from "@/lib/useMode";

export function HealthPill() {
  useModePolling();
  const mode = useStore((s) => s.mode);
  return <span className={`pill ${mode}`}>{mode === "live" ? "LIVE" : mode === "replay" ? "REPLAY" : "UNAVAILABLE"}</span>;
}
