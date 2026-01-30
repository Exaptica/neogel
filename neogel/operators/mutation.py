from __future__ import annotations

import numpy as np

from ..core.rng import RNG


def gaussian_mutation(x: np.ndarray, rng: RNG, sigma: float) -> np.ndarray:
    return x + rng.gen.normal(0.0, sigma, size=x.shape)