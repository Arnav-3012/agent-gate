// Mirror of engine/api/health.py: GET /health payload and the live/replay decision.
export interface Health {
  ok: boolean;
  llm: boolean;
  qdrant: boolean;
}

export type Mode = "live" | "replay" | "unavailable";

// health === null means the engine could not be reached at all.
export function decideMode(health: Health | null, replayAvailable: boolean): Mode {
  if (health && health.ok && health.llm && health.qdrant) return "live";
  return replayAvailable ? "replay" : "unavailable";
}
