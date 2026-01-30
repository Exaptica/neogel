from __future__ import annotations

import numpy as np
from neogel.core.types import Candidate


def steps_and_smoothness(c: Candidate) -> np.ndarray:
    if c.record is None or c.record.extras is None:
        raise ValueError("Missing extras for descriptor")

    steps = float(c.record.extras["steps"])
    smooth = float(c.record.extras["mean_abs_action"])

    d0 = np.clip(steps / 1600.0, 0.0, 1.0)
    d1 = np.clip(smooth, 0.0, 1.0)

    return np.array([d0, d1], dtype=float)