// Engine client. Live mode polls GET /trace/{id}?after=seq (D-plan: 500 ms); replay lists finished traces.
import type { TraceEvent } from "./schema";
import type { Health } from "./health";

export interface TraceView {
  trace_id: string;
  events: TraceEvent[];
  done: boolean;
}

export type Fetch = typeof fetch;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.url} -> ${res.status}`);
  return (await res.json()) as T;
}

export class EngineClient {
  constructor(private base: string, private f: Fetch = (...a) => fetch(...a)) {}

  run = async (question: string, arm?: string): Promise<string> =>
    (await json<{ trace_id: string }>(await this.f(`${this.base}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, ...(arm ? { arm } : {}) }),
    }))).trace_id;

  trace = async (id: string, after = -1): Promise<TraceView> =>
    json<TraceView>(await this.f(`${this.base}/trace/${id}?after=${after}`));

  replayIds = async (): Promise<string[]> => json<string[]>(await this.f(`${this.base}/replay`));

  // null = engine unreachable, which decideMode treats as "not live".
  health = async (): Promise<Health | null> => {
    try {
      return await json<Health>(await this.f(`${this.base}/health`));
    } catch {
      return null;
    }
  };
}

// Poll until the trace is done; `sleep` is injectable so tests need no real time.
export async function pollTrace(
  client: Pick<EngineClient, "trace">,
  id: string,
  onEvents: (events: TraceEvent[]) => void,
  { intervalMs = 500, maxPolls = 600, sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms)) } = {},
): Promise<boolean> {
  let after = -1;
  for (let i = 0; i < maxPolls; i++) {
    const view = await client.trace(id, after);
    if (view.events.length) {
      after = view.events[view.events.length - 1].seq;
      onEvents(view.events);
    }
    if (view.done) return true;
    await sleep(intervalMs);
  }
  return false;
}
