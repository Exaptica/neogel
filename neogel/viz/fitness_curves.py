from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def save_fitness_curve(
    *,
    generations: Sequence[int],
    best: Sequence[float],
    mean: Sequence[float] | None = None,
    out_path: str | Path,
    title: str = "Fitness over generations",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(generations, best, label="best")
    if mean is not None:
        plt.plot(generations, mean, label="mean")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()