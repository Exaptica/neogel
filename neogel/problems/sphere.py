from __future__ import annotations

import numpy as np

from neogel.core.types import EvalRecord


class SphereProblem:
    def evaluate(self, x: np.ndarray) -> EvalRecord:
        fitness = -float(np.sum(x * x))
        return EvalRecord(objectives=np.array([fitness], dtype=float), extras=None)