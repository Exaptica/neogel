from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.rng import RNG
from ..core.types import Candidate
from ..operators.selection import tournament_select
from ..operators.mutation import gaussian_mutation


@dataclass
class GAConfig:
    pop_size: int
    genome_dim: int
    init_low: float = -5.0
    init_high: float = 5.0
    mutation_sigma: float = 0.1
    elitism: int = 1


class GAEngine:
    def __init__(self, *, rng: RNG, cfg: GAConfig):
        self.rng = rng
        self.cfg = cfg
        self.population: list[Candidate] = []
        self._gen = 0
        self._best: float = -float("inf")
        self._mean: float = float("nan")

        # init pop (unevaluated; runner will evaluate before tell() is meaningful)
        for _ in range(cfg.pop_size):
            g = rng.gen.uniform(cfg.init_low, cfg.init_high, size=(cfg.genome_dim,))
            self.population.append(Candidate(genotype=g, record=None, meta={"age": 0}))

    def ask(self, n: int) -> list[Candidate]:
        # For generation 0: ask returns current population if not yet evaluated.
        if any(c.record is None for c in self.population):
            return self.population

        # Otherwise: create offspring
        parents = tournament_select(self.population, self.rng, n=n, k=3)
        offspring: list[Candidate] = []
        for p in parents:
            x = np.asarray(p.genotype)
            child = gaussian_mutation(x, self.rng, sigma=self.cfg.mutation_sigma)
            offspring.append(Candidate(genotype=child, record=None, meta={"age": 0}))
        return offspring

    def tell(self, evaluated: list[Candidate]) -> None:
        # evaluated is either initial population (gen0) or offspring
        fits = np.array([c.record.fitness_scalar for c in evaluated if c.record is not None], dtype=float)
        self._best = float(np.max(fits))
        self._mean = float(np.mean(fits))

        # If we're at gen0 initialization, store evaluated population
        if any(c.record is None for c in self.population):
            self.population = evaluated
            self._gen += 1
            return

        # Elitist generational replacement
        elites = sorted(self.population, key=lambda c: c.record.fitness_scalar, reverse=True)[: self.cfg.elitism]
        rest = sorted(evaluated, key=lambda c: c.record.fitness_scalar, reverse=True)[: self.cfg.pop_size - self.cfg.elitism]
        self.population = elites + rest
        for c in self.population:
            if c.meta is None:
                c.meta = {}
            c.meta["age"] = int(c.meta.get("age", 0)) + 1

        self._gen += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "gen": self._gen,
            "best_fitness": self._best,
            "mean_fitness": self._mean,
            "pop_size": len(self.population),
        }