from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from neogel.core.types import Candidate


@dataclass(slots=True)
class ArchiveGridSpec:
    mins: np.ndarray   # (d,)
    maxs: np.ndarray   # (d,)
    bins: np.ndarray   # (d,)


class ArchiveGrid:
    """Standard MAP-Elites grid archive (maximization)."""

    def __init__(self, spec: ArchiveGridSpec):
        self.spec = spec
        self.dim = int(spec.bins.size)

        self.shape = tuple(int(b) for b in spec.bins.tolist())
        self.fitness = np.full(self.shape, -np.inf, dtype=float)
        self.occupied = np.zeros(self.shape, dtype=bool)
        self.elites: Dict[Tuple[int, ...], Candidate] = {}

    def _cell_index(self, desc: np.ndarray) -> Tuple[int, ...] | None:
        desc = np.asarray(desc, dtype=float)

        if np.any(desc < self.spec.mins) or np.any(desc > self.spec.maxs):
            return None

        frac = (desc - self.spec.mins) / (self.spec.maxs - self.spec.mins + 1e-12)
        idx = np.floor(frac * self.spec.bins).astype(int)
        idx = np.clip(idx, 0, self.spec.bins - 1)

        return tuple(int(i) for i in idx.tolist())

    def add(self, cand: Candidate, desc: np.ndarray) -> bool:
        if cand.record is None:
            raise ValueError("Cannot insert unevaluated candidate")

        idx = self._cell_index(desc)
        if idx is None:
            return False

        fit = cand.record.fitness_scalar
        if (not self.occupied[idx]) or fit > self.fitness[idx]:
            self.occupied[idx] = True
            self.fitness[idx] = fit
            self.elites[idx] = cand
            return True

        return False

    def sample(self, n: int, rng: np.random.Generator) -> list[Candidate]:
        if not self.elites:
            return []
        keys = list(self.elites.keys())
        idxs = rng.choice(len(keys), size=n, replace=True)
        return [self.elites[keys[int(i)]] for i in idxs]

    # ---- metrics ----
    def coverage(self) -> float:
        return float(np.mean(self.occupied))

    def qd_score(self) -> float:
        f = self.fitness.copy()
        f[~self.occupied] = 0.0
        return float(np.sum(f))

    def max_fitness(self) -> float:
        if not np.any(self.occupied):
            return -float("inf")
        return float(np.max(self.fitness[self.occupied]))