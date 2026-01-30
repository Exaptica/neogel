from __future__ import annotations

import numpy as np

from neogel.core.types import EvalRecord


class SphereQDProblem:
    """Sphere fitness with 2D descriptor = first two genotype coords (clipped to [-1, 1])."""

    def __init__(self, clip: float = 1.0):
        self.clip = float(clip)

    def evaluate(self, x: np.ndarray) -> EvalRecord:
        x = np.asarray(x, dtype=float)

        fitness = -float(np.sum(x * x))

        d0 = float(np.clip(x[0], -self.clip, self.clip))
        d1 = float(np.clip(x[1], -self.clip, self.clip))

        extras = {"d0": d0, "d1": d1}
        return EvalRecord(objectives=np.array([fitness], dtype=float), extras=extras)