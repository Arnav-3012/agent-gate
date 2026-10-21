// Shape of eval/results.json (written by eval/benchmark.py).
export interface GroupStat { f1: number; n: number }

export interface ArmStats {
  f1: number;
  f1_ci: [number, number];
  em: number;
  cost_per_1k: number;
  latency_p50: number;
  latency_p95: number;
  by_type: Record<string, GroupStat>;
  by_level: Record<string, GroupStat>;
  n: number;
}

export interface OracleRouter {
  f1: number;
  f1_ci: [number, number];
  cost_per_1k: number;
  gain_over_best_fixed: number;
}

export interface DomainResults {
  n: number;
  arms: Record<string, ArmStats>;
  oracle_router?: OracleRouter;
}

export type Results = Record<string, DomainResults>;

const arm = (f1: number, cost: number, p50: number, p95: number): ArmStats => ({
  f1, f1_ci: [f1 - 0.06, f1 + 0.06], em: f1 * 0.7, cost_per_1k: cost, latency_p50: p50, latency_p95: p95,
  by_type: { bridge: { f1: f1 - 0.05, n: 20 }, comparison: { f1: f1 + 0.05, n: 20 } },
  by_level: { easy: { f1: f1 + 0.1, n: 13 }, hard: { f1: f1 - 0.1, n: 14 } }, n: 40,
});

// Synthetic placeholder numbers, only for developing the page. Never shown as results.
export const MOCK_RESULTS: Results = {
  general: {
    n: 40,
    arms: {
      "qwen7b-rag": arm(0.5, 0.08, 1.2, 2.4),
      "qwen7b-agent": arm(0.52, 0.31, 6.5, 14),
      "qwen14b-rag": arm(0.58, 0.16, 2.4, 4.6),
      "qwen14b-agent": arm(0.6, 0.62, 13, 28),
    },
    oracle_router: { f1: 0.7, f1_ci: [0.64, 0.76], cost_per_1k: 0.2, gain_over_best_fixed: 0.1 },
  },
};
