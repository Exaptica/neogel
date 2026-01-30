from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.types import Candidate
from ..core.rng import RNG


def tournament_select(pop: Sequence[Candidate], rng: RNG, n: int, k: int = 3) -> list[Candidate]:
    """Select n parents via k-tournament on single-objective fitness."""
    pop_list = list(pop)
    idxs = np.arange(len(pop_list))
    chosen: list[Candidate] = []

    for _ in range(n):
        contenders = rng.gen.choice(idxs, size=k, replace=False)
        best = None
        best_fit = -float("inf")
        for j in contenders:
            r = pop_list[int(j)].record
            if r is None:
                raise ValueError("Population contains unevaluated candidate.")
            f = r.fitness_scalar
            if f > best_fit:
                best_fit = f
                best = pop_list[int(j)]
        chosen.append(best)  # type: ignore[arg-type]
    return chosen