"""Disjoint LinUCB contextual bandit: one ridge regression per arm, pick argmax of mean + alpha*width.
Reward is whatever the caller defines; `reward()` is the F1-gain-minus-cost form used in the plan."""
import numpy as np


def reward(score: float, cost: float, cost_weight: float) -> float:
    return score - cost_weight * cost


class LinUCB:
    def __init__(self, arms: list[str], dim: int, alpha: float = 1.0, seed: int = 0) -> None:
        self.arms, self.alpha = arms, alpha
        self._a = {k: np.eye(dim) for k in arms}  # X^T X + I
        self._b = {k: np.zeros(dim) for k in arms}  # X^T r
        self._rng = np.random.default_rng(seed)

    def ucb(self, arm: str, x: np.ndarray) -> float:
        inv = np.linalg.inv(self._a[arm])
        return float((inv @ self._b[arm]) @ x + self.alpha * np.sqrt(x @ inv @ x))

    def choose(self, x: np.ndarray) -> str:
        scores = np.array([self.ucb(k, x) for k in self.arms])
        return self.arms[int(self._rng.choice(np.flatnonzero(scores == scores.max())))]

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        self._a[arm] += np.outer(x, x)
        self._b[arm] += r * x
