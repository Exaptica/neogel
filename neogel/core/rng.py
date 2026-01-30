from __future__ import annotations

import numpy as np


class RNG:
    """Small wrapper to keep RNG explicit and consistent across the codebase."""
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._gen = np.random.default_rng(self.seed)

    @property
    def gen(self) -> np.random.Generator:
        return self._gen