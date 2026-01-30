import numpy as np

from neogel.core.types import EvalRecord
from neogel.core.rng import RNG
from neogel.core.evaluator import SerialEvaluator
from neogel.core.runner import Runner, RunConfig
from neogel.core.run_manager import RunManager
from neogel.engines.ga import GAEngine, GAConfig
from neogel.logging.sinks import RichSink, CSVSink, PlotSink

class SphereProblem:
    def evaluate(self, x: np.ndarray) -> EvalRecord:
        # maximize negative L2 (so closer to 0 is better)
        fitness = -float(np.sum(x * x))
        return EvalRecord(objectives=np.array([fitness], dtype=float), extras=None)


def transform(payload: dict) -> dict:
    """Customize what goes into metrics.csv."""
    e = payload.get("engine", {})
    return {
        "gen": payload.get("gen"),
        "elapsed_sec": payload.get("elapsed_sec"),
        "evals_total": payload.get("evals_total"),
        "best_fitness": e.get("best_fitness"),
        "mean_fitness": e.get("mean_fitness"),
    }


if __name__ == "__main__":
    seed = 0
    rng = RNG(seed=seed)

    rm = RunManager(root="runs")
    paths = rm.create(experiment="ga_sphere", seed=seed)

    engine = GAEngine(rng=rng, cfg=GAConfig(pop_size=200, genome_dim=20, mutation_sigma=0.15))

    runner = Runner(
        engine=engine,
        problem=SphereProblem(),
        evaluator=SerialEvaluator(),
        sinks=[
            RichSink(),
            CSVSink(paths.metrics_csv, transform=transform, flatten=False),
            PlotSink(paths.artifacts_dir),
        ],
        cfg=RunConfig(generations=200, pop_size=200, log_every=1),
    )

    out = runner.run()
    print("\nRun saved to:", paths.run_dir)
    print("DONE:", out)