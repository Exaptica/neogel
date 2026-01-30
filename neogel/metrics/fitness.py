from __future__ import annotations
from typing import Any
import numpy as np

from neogel.core.types import Candidate


def best_mean_std(population: list[Candidate], **_) -> dict[str, Any]:
    fits = np.array([c.record.fitness_scalar for c in population if c.record is not None], dtype=float)
    return {
        "best_fitness": float(np.max(fits)) if len(fits) else None,
        "mean_fitness": float(np.mean(fits)) if len(fits) else None,
        "std_fitness": float(np.std(fits)) if len(fits) else None,
    }