from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from neogel.core.config import load_config
from neogel.core.run_manager import RunManager
from neogel.core.rng import RNG
from neogel.core.evaluator import SerialEvaluator, ProcessPoolEvaluator
from neogel.core.runner import Runner, RunConfig
from neogel.logging.sinks import RichSink, CSVSink, JSONLSink
from neogel.core.imports import import_by_path
from neogel.core.build import build_engine, build_problem, build_sinks


def main() -> None:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    p = argparse.ArgumentParser(prog="neogel", description="neogel experiment runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run an experiment from a YAML config")
    run.add_argument("config", type=str, help="path to config yaml (e.g., configs/ga/sphere.yaml)")
    run.add_argument("--seed", type=int, default=None, help="override seed in config")
    run.add_argument("--workers", type=int, default=None, help="use ProcessPool with N workers")
    run.add_argument("--tag", type=str, default=None, help="optional run tag suffix")
    run.add_argument("--runs-root", type=str, default="runs", help="root folder for run outputs")
    run.add_argument("--no-rich", action="store_true", help="disable rich terminal sink")

    args = p.parse_args()

    if args.cmd == "run":
        cfg = load_config(args.config)

        seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
        exp_name = str(cfg.get("experiment", Path(args.config).stem))

        rm = RunManager(root=args.runs_root)
        paths = rm.create(experiment=exp_name, seed=seed, tag=args.tag)

        rng = RNG(seed=seed)

        # Evaluator
        workers = args.workers if args.workers is not None else cfg.get("workers", None)
        evaluator = ProcessPoolEvaluator(max_workers=int(workers)) if workers else SerialEvaluator()

        engine = build_engine(cfg["engine"], rng=rng)
        problem = build_problem(cfg["problem"])

        run_cfg = RunConfig(
            generations=int(cfg["run"]["generations"]),
            pop_size=int(cfg["run"]["pop_size"]),
            log_every=int(cfg["run"].get("log_every", 1)),
        )

        sinks = build_sinks(
            cfg.get("logging", {}),
            paths=paths,
            enable_rich=not args.no_rich,
        )

        runner = Runner(engine=engine, problem=problem, evaluator=evaluator, sinks=sinks, cfg=run_cfg)
        out = runner.run()

        print("\nRun saved to:", paths.run_dir)
        print("DONE:", out)


if __name__ == "__main__":
    main()