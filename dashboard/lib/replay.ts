// Replay engine: plays recorded events at their original spacing (scaled), same path as live polling.
import type { TraceEvent } from "./schema";

export interface Scheduler {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (h: unknown) => void;
}

const realScheduler: Scheduler = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export interface ReplayOptions {
  speed?: number; // 2 = twice as fast
  maxGapMs?: number; // long LLM calls are capped so the demo does not stall
  scheduler?: Scheduler;
  onDone?: () => void;
}

// Delay before event i, from the recorded timestamps (seconds).
export function gaps(events: TraceEvent[], speed = 1, maxGapMs = 1500): number[] {
  return events.map((e, i) =>
    i === 0 ? 0 : Math.min(Math.max(0, (e.ts - events[i - 1].ts) * 1000) / speed, maxGapMs),
  );
}

// Returns a cancel function.
export function play(events: TraceEvent[], onEvent: (e: TraceEvent) => void, opts: ReplayOptions = {}): () => void {
  const { speed = 1, maxGapMs = 1500, scheduler = realScheduler, onDone } = opts;
  const delays = gaps(events, speed, maxGapMs);
  let handle: unknown;
  let cancelled = false;
  const step = (i: number) => {
    if (cancelled) return;
    if (i >= events.length) return onDone?.();
    handle = scheduler.setTimeout(() => {
      if (cancelled) return;
      onEvent(events[i]);
      step(i + 1);
    }, delays[i]);
  };
  step(0);
  return () => {
    cancelled = true;
    scheduler.clearTimeout(handle);
  };
}
