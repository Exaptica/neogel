from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from neogel.core.types import Candidate
from neogel.core.rng import RNG
from neogel.containers.archive_grid import ArchiveGrid


@dataclass
class GaussianEmitterConfig:
    genome_dim: int
    init_low: float = -1.0
    init_high: float = 1.0
    sigma: float = 0.1
    init_fraction: float = 0.2


class GaussianEmitter:
    """Standard MAP-Elites Gaussian emitter."""

    def __init__(self, *, rng: RNG, cfg: GaussianEmitterConfig, archive: ArchiveGrid):
        self.rng = rng
        self.cfg = cfg
        self.archive = archive

    def emit(self, n: int) -> list[Candidate]:
        n_init = int(self.cfg.init_fraction * n)
        n_mut = n - n_init

        batch: list[Candidate] = []

        # Random initialization
        for _ in range(n_init):
            g = self.rng.gen.uniform(
                self.cfg.init_low, self.cfg.init_high, size=(self.cfg.genome_dim,)
            )
            batch.append(Candidate(genotype=g, record=None, meta={"src": "init"}))

        # Mutate elites
        elites = self.archive.sample(n_mut, rng=self.rng.gen)
        if elites:
            for e in elites:
                x = np.asarray(e.genotype)
                child = x + self.rng.gen.normal(0.0, self.cfg.sigma, size=x.shape)
                batch.append(Candidate(genotype=child, record=None, meta={"src": "elite"}))
        else:
            for _ in range(n_mut):
                g = self.rng.gen.uniform(
                    self.cfg.init_low, self.cfg.init_high, size=(self.cfg.genome_dim,)
                )
                batch.append(Candidate(genotype=g, record=None, meta={"src": "fallback"}))

        return batch