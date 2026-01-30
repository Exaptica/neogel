from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Protocol
import json
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from ..viz.fitness_curves import save_fitness_curve

from neogel.viz.fitness_curves import save_fitness_curve
from neogel.viz.archive_heatmap import save_archive_heatmap

class Sink(Protocol):
    def log(self, payload: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class RichSink:
    def __init__(self):
        self.console = Console()
        self._live: Live | None = None
        self._last_payload: dict[str, Any] | None = None

    def _render(self, payload: dict[str, Any]) -> Table:
        t = Table(title="neogel run", show_lines=False)
        t.add_column("key", style="bold")
        t.add_column("value")

        t.add_row("generation", str(payload.get("gen")))
        engine = payload.get("engine", {})
        if isinstance(engine, dict):
            for k, v in engine.items():
                t.add_row(str(k), str(v))
        return t

    def log(self, payload: dict[str, Any]) -> None:
        self._last_payload = payload
        if self._live is None:
            self._live = Live(self._render(payload), console=self.console, refresh_per_second=8)
            self._live.__enter__()
        else:
            self._live.update(self._render(payload))

    def close(self) -> None:
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None


def _flatten_dict(d: dict[str, Any], *, sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    stack: list[tuple[str, Any]] = [("", d)]
    while stack:
        prefix, obj = stack.pop()
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                stack.append((key, v))
        else:
            # prefix is "" only if d wasn't a dict (not possible here), but keep safe.
            out[prefix or "value"] = obj
    return out


class CSVSink:
    """Writes one row per log() call.

    Customization:
    - transform(payload) -> dict: choose what to log and shape it
    - flatten=True: auto-flatten nested dicts using sep
    - fieldnames: fix a column order (otherwise inferred dynamically)

    Note:
    - If you don't provide `fieldnames`, columns are inferred from the first row.
      Later fields not present in the first row will be ignored (extrasaction="ignore").
    """

    def __init__(
        self,
        path: str | Path,
        *,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        flatten: bool = True,
        sep: str = ".",
        fieldnames: list[str] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.transform = transform
        self.flatten = flatten
        self.sep = sep
        self.fieldnames = fieldnames

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None

    def _ensure_writer(self, row: dict[str, Any]) -> None:
        if self._writer is not None:
            return

        if self.fieldnames is None:
            fns = list(row.keys())  # inferred from first row
        else:
            fns = list(self.fieldnames)

        self._writer = csv.DictWriter(self._file, fieldnames=fns, extrasaction="ignore")
        self._writer.writeheader()

    def log(self, payload: dict[str, Any]) -> None:
        row = self.transform(payload) if self.transform else dict(payload)
        if self.flatten:
            row = _flatten_dict(row, sep=self.sep)

        self._ensure_writer(row)
        self._writer.writerow(row)  # type: ignore[union-attr]
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

class PlotSink:
    """Buffers metrics during a run and writes plots on close().

    Writes:
      - fitness_curve.png (best/mean if available)
      - archive_heatmap.png if engine has a 2D ArchiveGrid at engine.archive
    """

    def __init__(
        self,
        artifacts_dir: str | Path,
        *,
        extract_curve: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        curve_filename: str = "fitness_curve.png",
        heatmap_filename: str = "archive_heatmap.png",
        curve_title: str = "Fitness over generations",
        heatmap_title: str = "MAP-Elites Archive (Fitness Heatmap)",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.extract_curve = extract_curve
        self.curve_filename = curve_filename
        self.heatmap_filename = heatmap_filename
        self.curve_title = curve_title
        self.heatmap_title = heatmap_title

        self._gens: list[int] = []
        self._best: list[float] = []
        self._mean: list[float] = []

        # We keep the last engine reference we saw so we can read archive on close().
        self._last_engine_obj: Any | None = None

    def log(self, payload: dict[str, Any]) -> None:
        # keep a handle to the engine object if it's in payload (optional)
        if "engine_obj" in payload:
            self._last_engine_obj = payload["engine_obj"]

        # default extraction: try GA-style keys first, then MAP-Elites max_fitness
        if self.extract_curve is None:
            e = payload.get("engine", {})  # engine.metrics() dict
            gen = payload.get("gen")

            # Prefer GA metrics if present
            best = e.get("best_fitness")
            mean = e.get("mean_fitness")

            # For MAP-Elites, use max_fitness as "best"
            if best is None:
                best = e.get("max_fitness")

            row = {"gen": gen, "best": best, "mean": mean}
        else:
            row = self.extract_curve(payload)

        g = row.get("gen")
        b = row.get("best")
        m = row.get("mean")

        if g is None or b is None:
            return

        self._gens.append(int(g))
        self._best.append(float(b))
        if m is not None:
            self._mean.append(float(m))

    def close(self) -> None:
        # --- Fitness curve ---
        if self._gens:
            out_path = self.artifacts_dir / self.curve_filename
            mean = self._mean if len(self._mean) == len(self._gens) else None
            save_fitness_curve(
                generations=self._gens,
                best=self._best,
                mean=mean,
                out_path=out_path,
                title=self.curve_title,
            )

        # --- Archive heatmap (MAP-Elites) ---
        # We try to find a live archive object. Preferred: payload provides engine_obj.
        engine = self._last_engine_obj
        if engine is None:
            return

        archive = getattr(engine, "archive", None)
        if archive is None:
            return

        # Only for 2D archives
        try:
            if len(getattr(archive, "shape", ())) != 2:
                return
        except Exception:
            return

        out_path = self.artifacts_dir / self.heatmap_filename
        save_archive_heatmap(archive, out_path, title=self.heatmap_title)


class JSONLSink:
    """Writes one JSON object per log() call (newline-delimited JSON).

    Great for nested metrics and future QD archive stats.
    """
    def __init__(self, path: str | Path, *, transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.transform = transform
        self._file = open(self.path, "w", encoding="utf-8")

    def log(self, payload: dict[str, Any]) -> None:
        row = self.transform(payload) if self.transform else payload
        # Add a timestamp for convenience (doesn’t break anything if you ignore it).
        if "ts" not in row:
            row = dict(row)
            row["ts"] = datetime.now().isoformat(timespec="seconds")
        self._file.write(json.dumps(row, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass