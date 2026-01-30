from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .types import Candidate, Engine, Evaluator, Problem
from ..logging.sinks import Sink


@dataclass(slots=True)
class RunConfig:
    generations: int
    pop_size: int
    log_every: int = 1


MetricFn = Callable[..., dict[str, Any]]


class Runner:
    def __init__(
        self,
        *,
        engine: Engine,
        problem: Problem,
        evaluator: Evaluator,
        sinks: list[Sink] | None = None,
        cfg: RunConfig,
        metrics: list[MetricFn] | None = None,
    ):
        self.engine = engine
        self.problem = problem
        self.evaluator = evaluator
        self.sinks = sinks or []
        self.cfg = cfg
        self.metrics = metrics or []

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        evals_total = 0

        for gen in range(self.cfg.generations):
            # 1) ask
            batch = self.engine.ask(self.cfg.pop_size)
            genotypes = [c.genotype for c in batch]

            # 2) evaluate (serial or parallel via evaluator)
            records = self.evaluator.map(self.problem.evaluate, genotypes)

            evaluated: list[Candidate] = []
            for c, r in zip(batch, records):
                c.record = r
                evaluated.append(c)

            evals_total += len(evaluated)

            # 3) tell
            self.engine.tell(evaluated)

            # 4) log
            if (gen % self.cfg.log_every) == 0:
                elapsed_sec = time.time() - t0

                # Collect custom metrics (safe: metric failures won't crash the run)
                custom: dict[str, Any] = {}
                for m in self.metrics:
                    try:
                        out = m(
                            engine=self.engine,
                            batch=evaluated,
                            gen=gen,
                            elapsed_sec=elapsed_sec,
                            evals_total=evals_total,
                        )
                        if out:
                            custom.update(out)
                    except Exception as e:
                        custom[f"metric_error.{getattr(m, '__name__', 'metric')}"] = str(e)

                payload = {
                    "gen": gen,
                    "elapsed_sec": elapsed_sec,
                    "evals_total": evals_total,
                    "engine": self.engine.metrics(),
                    "engine_obj": self.engine,
                    "metrics": custom,
                }
                for s in self.sinks:
                    s.log(payload)

        # close sinks
        for s in self.sinks:
            s.close()

        return {
            "walltime_sec": time.time() - t0,
            "evals_total": evals_total,
            "final": self.engine.metrics(),
        }