"""Offline replay of a bandit over logged per-arm rewards (every arm was run on every question, so
the reward of whichever arm the bandit picks is known). Reports cumulative regret vs the best arm
per question and the mean reward of the policy."""
import numpy as np

from engine.router.bandit import LinUCB


def replay(bandit: LinUCB, X: np.ndarray, R: np.ndarray, order: np.ndarray | None = None) -> dict:
    """X: (n, d) features; R: (n, n_arms) rewards in bandit.arms order."""
    order = np.arange(len(X)) if order is None else order
    regret, picks = [], []
    for i in order:
        k = bandit.arms.index(bandit.choose(X[i]))
        bandit.update(bandit.arms[k], X[i], float(R[i, k]))
        regret.append(float(R[i].max() - R[i, k]))
        picks.append(k)
    return {"regret": np.cumsum(regret), "mean_reward": float(np.mean([R[i, k] for i, k in zip(order, picks, strict=True)])),
            "picks": picks}
