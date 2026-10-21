"use client";
import { useEffect } from "react";
import { client } from "./engine";
import { decideMode } from "./health";
import { useStore } from "./store";

// Polls /health; recorded traces (bundled mock for now) count as the replay fallback.
export function useModePolling(intervalMs = 3000, replayAvailable = true) {
  const setMode = useStore((s) => s.setMode);
  useEffect(() => {
    let live = true;
    const tick = async () => {
      const h = await client.health();
      if (live) setMode(decideMode(h, replayAvailable));
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { live = false; clearInterval(id); };
  }, [intervalMs, replayAvailable, setMode]);
}
