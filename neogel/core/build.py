from __future__ import annotations

from typing import Any

import numpy as np

from neogel.core.imports import import_by_path
from neogel.core.rng import RNG
from neogel.engines.ga import GAEngine, GAConfig

from neogel.logging.sinks import RichSink, CSVSink, JSONLSink, PlotSink

# QD / MAP-Elites
from neogel.containers.archive_grid import ArchiveGrid, ArchiveGridSpec
from neogel.emitters.gaussian_emitter import GaussianEmitter, GaussianEmitterConfig
from neogel.engines.mapelites import MapElitesEngine


def build_problem(problem_cfg: dict[str, Any]):
    cls = import_by_path(problem_cfg["path"])
    return cls()  # v0.1: no-arg constructor


def build_engine(engine_cfg: dict[str, Any], *, rng: RNG):
    name = engine_cfg["name"]
    params = engine_cfg.get("params", {})

    # ---- GA ----
    if name == "ga":
        cfg = GAConfig(**params)
        return GAEngine(rng=rng, cfg=cfg)

    # ---- MAP-Elites (normal QD) ----
    if name == "mapelites":
        # Archive
        arch_cfg = engine_cfg["archive"]
        spec = ArchiveGridSpec(
            mins=np.array(arch_cfg["mins"], dtype=float),
            maxs=np.array(arch_cfg["maxs"], dtype=float),
            bins=np.array(arch_cfg["bins"], dtype=int),
        )
        archive = ArchiveGrid(spec)

        # Descriptor function
        desc_path = engine_cfg["descriptor"]["path"]
        descriptor_fn = import_by_path(desc_path)

        # Emitter (Gaussian)
        em_cfg = engine_cfg["emitter"]["params"]
        emitter_cfg = GaussianEmitterConfig(**em_cfg)
        emitter = GaussianEmitter(rng=rng, cfg=emitter_cfg, archive=archive)

        return MapElitesEngine(rng=rng, archive=archive, emitter=emitter, descriptor_fn=descriptor_fn)

    raise ValueError(f"Unknown engine name: {name}")


def build_sinks(logging_cfg: dict[str, Any], *, paths, enable_rich: bool = True):
    sinks = []

    if enable_rich and logging_cfg.get("rich", {}).get("enabled", True):
        sinks.append(RichSink())

    if logging_cfg.get("csv", {}).get("enabled", True):
        sinks.append(
            CSVSink(
                paths.metrics_csv,
                flatten=bool(logging_cfg.get("csv", {}).get("flatten", True)),
                sep=str(logging_cfg.get("csv", {}).get("sep", ".")),
            )
        )

    if logging_cfg.get("jsonl", {}).get("enabled", True):
        sinks.append(JSONLSink(paths.run_dir / "metrics.jsonl"))

    if logging_cfg.get("plots", {}).get("enabled", True):
        sinks.append(PlotSink(paths.artifacts_dir))

    return sinks