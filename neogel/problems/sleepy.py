from __future__ import annotations

import time
import numpy as np

from neogel.core.types import EvalRecord


class SleepySphereProblem:
    """Sphere, but sleeps per evaluation to make parallel speedups obvious."""

    def __init__(self, sleep_sec: float = 0.01):
        self.sleep_sec = float(sleep_sec)

    def evaluate(self, x: np.ndarray) -> EvalRecord:
        time.sleep(self.sleep_sec)
        fitness = -float(np.sum(x * x))
        return EvalRecord(objectives=np.array([fitness], dtype=float), extras=None)