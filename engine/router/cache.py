"""Semantic cache: reuse a past answer when a new query is a near-duplicate (cosine >= threshold)."""
import time
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel

CACHE_ROUTE = "cache_hit"


class CacheEntry(BaseModel):
    query: str
    answer: str
    arm: str  # arm that originally produced the answer
    stored_at: float


class SemanticCache:
    """embed must return L2-normalised vectors (the bge embedder does), so dot product = cosine."""

    def __init__(self, embed: Callable[[str], list[float]], threshold: float = 0.92,
                 ttl_s: float | None = 3600, clock: Callable[[], float] = time.time) -> None:
        self.embed, self.threshold, self.ttl_s, self.clock = embed, threshold, ttl_s, clock
        self._vecs: list[np.ndarray] = []
        self._entries: list[CacheEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, query: str, answer: str, arm: str) -> None:
        self._vecs.append(np.asarray(self.embed(query), dtype=float))
        self._entries.append(CacheEntry(query=query, answer=answer, arm=arm, stored_at=self.clock()))

    def get(self, query: str) -> tuple[CacheEntry, float] | None:
        """Best live entry at or above the threshold, with its similarity; None on a miss."""
        if not self._entries:
            return None
        sims = np.stack(self._vecs) @ np.asarray(self.embed(query), dtype=float)
        now = self.clock()
        for i in np.argsort(-sims):
            if sims[i] < self.threshold:
                return None
            if self.ttl_s is None or now - self._entries[i].stored_at <= self.ttl_s:
                return self._entries[i], float(sims[i])
        return None
