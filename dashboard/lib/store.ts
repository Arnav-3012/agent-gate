// zustand store: one active trace, fed by live polling or replay through the same reducer.
import { create } from "zustand";
import type { TraceEvent } from "./schema";
import { emptyState, reduce, type TraceState } from "./trace";
import type { Mode } from "./health";

interface Store {
  mode: Mode;
  trace: TraceState;
  setMode: (m: Mode) => void;
  push: (events: TraceEvent[]) => void;
  reset: () => void;
}

export const useStore = create<Store>((set) => ({
  mode: "unavailable",
  trace: emptyState(),
  setMode: (mode) => set({ mode }),
  push: (events) => set((s) => ({ trace: events.reduce(reduce, s.trace) })),
  reset: () => set({ trace: emptyState() }),
}));
