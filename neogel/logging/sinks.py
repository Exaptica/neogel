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
    """Buffers metrics and writes plots on close().

    Customization:
    - extract(payload) -> dict with keys:
        - gen (int)
        - best (float)
        - mean (float) optional
    """
    def __init__(
        self,
        artifacts_dir: str | Path,
        *,
        extract: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        filename: str = "fitness_curve.png",
        title: str = "Fitness over generations",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.extract = extract
        self.filename = filename
        self.title = title

        self._gens: list[int] = []
        self._best: list[float] = []
        self._mean: list[float] = []

    def log(self, payload: dict[str, Any]) -> None:
        if self.extract is None:
            # default assumes Runner payload + GA metrics
            e = payload.get("engine", {})
            row = {
                "gen": payload.get("gen"),
                "best": e.get("best_fitness"),
                "mean": e.get("mean_fitness"),
            }
        else:
            row = self.extract(payload)

        g = row.get("gen")
        b = row.get("best")
        m = row.get("mean")

        if g is None or b is None:
            return  # ignore incomplete rows

        self._gens.append(int(g))
        self._best.append(float(b))
        if m is not None:
            self._mean.append(float(m))

    def close(self) -> None:
        if not self._gens:
            return
        out_path = self.artifacts_dir / self.filename

        mean = self._mean if len(self._mean) == len(self._gens) else None
        save_fitness_curve(
            generations=self._gens,
            best=self._best,
            mean=mean,
            out_path=out_path,
            title=self.title,
        )

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