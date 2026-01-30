from __future__ import annotations

from typing import Callable, Any

import numpy as np

from neogel.core.types import Candidate
from neogel.core.rng import RNG
from neogel.containers.archive_grid import ArchiveGrid
from neogel.emitters.gaussian_emitter import GaussianEmitter


DescriptorFn = Callable[[Candidate], np.ndarray]


class MapElitesEngine:
    """Standard MAP-Elites engine."""

    def __init__(
        self,
        *,
        rng: RNG,
        archive: ArchiveGrid,
        emitter: GaussianEmitter,
        descriptor_fn: DescriptorFn,
    ):
        self.rng = rng
        self.archive = archive
        self.emitter = emitter
        self.descriptor_fn = descriptor_fn

        self._gen = 0
        self._last_inserts = 0

    def ask(self, n: int) -> list[Candidate]:
        return self.emitter.emit(n)

    def tell(self, evaluated: list[Candidate]) -> None:
        inserts = 0
        for c in evaluated:
            desc = self.descriptor_fn(c)
            if self.archive.add(c, desc):
                inserts += 1
        self._last_inserts = inserts
        self._gen += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "gen": self._gen,
            "coverage": self.archive.coverage(),
            "qd_score": self.archive.qd_score(),
            "max_fitness": self.archive.max_fitness(),
            "inserts": self._last_inserts,
        }