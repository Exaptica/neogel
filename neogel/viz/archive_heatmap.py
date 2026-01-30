from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from neogel.containers.archive_grid import ArchiveGrid


def save_archive_heatmap(
    archive: ArchiveGrid,
    out_path: str | Path,
    *,
    title: str = "MAP-Elites Archive (Fitness Heatmap)",
    xlabel: str = "Descriptor dim 1 (bin)",
    ylabel: str = "Descriptor dim 0 (bin)",
):
    """Save a fitness heatmap of a 2D ArchiveGrid.

    - Occupied cells are colored by fitness.
    - Unoccupied cells are shown as NaN (blank).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(archive.shape) != 2:
        raise ValueError(f"Heatmap currently supports 2D archives only, got shape={archive.shape}")

    data = archive.fitness.copy()
    data[~archive.occupied] = np.nan

    plt.figure()
    plt.imshow(data, origin="lower", aspect="auto")
    plt.colorbar(label="fitness")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()