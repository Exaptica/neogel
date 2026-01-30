from __future__ import annotations

import numpy as np

from neogel.core.types import Candidate


def first2_coords(c: Candidate) -> np.ndarray:
    if c.record is None or c.record.extras is None:
        raise ValueError("Missing extras for descriptor.")
    return np.array([float(c.record.extras["d0"]), float(c.record.extras["d1"])], dtype=float)