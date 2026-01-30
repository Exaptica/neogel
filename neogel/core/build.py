from __future__ import annotations

from typing import Any, Callable

from neogel.core.imports import import_by_path
from neogel.core.rng import RNG
from neogel.engines.ga import GAEngine, GAConfig
from neogel.logging.sinks import RichSink, CSVSink, JSONLSink, PlotSink


def build_problem(problem_cfg: dict[str, Any]):
    cls = import_by_path(problem_cfg["path"])
    return cls()  # expects a no-arg constructor for v0.1


def build_engine(engine_cfg: dict[str, Any], *, rng: RNG):
    name = engine_cfg["name"]
    params = engine_cfg.get("params", {})

    if name == "ga":
        cfg = GAConfig(**params)
        return GAEngine(rng=rng, cfg=cfg)

    raise ValueError(f"Unknown engine name: {name}")


def build_sinks(logging_cfg: dict[str, Any], *, paths, enable_rich: bool = True):
    sinks = []

    if enable_rich and logging_cfg.get("rich", {}).get("enabled", True):
        sinks.append(RichSink())

    if logging_cfg.get("csv", {}).get("enabled", True):
        sinks.append(CSVSink(paths.metrics_csv, flatten=bool(logging_cfg.get("csv", {}).get("flatten", True))))

    if logging_cfg.get("jsonl", {}).get("enabled", True):
        sinks.append(JSONLSink(paths.run_dir / "metrics.jsonl"))

    # ✅ Add this
    if logging_cfg.get("plots", {}).get("enabled", True):
        sinks.append(PlotSink(paths.artifacts_dir))

    return sinks