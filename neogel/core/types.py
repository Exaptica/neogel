from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np


FitnessLike = float | Sequence[float] | np.ndarray
EvaluateFn = Callable[[Any], "EvalRecord"]


@dataclass(slots=True)
class EvalRecord:
    """Normalized evaluation result.

    - objectives: shape (m,) where m=1 for single-objective
    - extras: optional additional arrays/scalars for logging/analysis
    """
    objectives: np.ndarray
    extras: dict[str, Any] | None = None

    @property
    def is_multiobjective(self) -> bool:
        return self.objectives.size > 1

    @property
    def fitness_scalar(self) -> float:
        """Convenience scalar for single-objective problems."""
        if self.objectives.size != 1:
            raise ValueError("fitness_scalar is only valid for single-objective records.")
        return float(self.objectives[0])


@dataclass(slots=True)
class Candidate:
    genotype: Any
    record: EvalRecord | None = None
    meta: dict[str, Any] | None = None


@runtime_checkable
class Problem(Protocol):
    def evaluate(self, genotype: Any) -> EvalRecord: ...


@runtime_checkable
class Evaluator(Protocol):
    def map(self, fn: EvaluateFn, items: Sequence[Any]) -> list[EvalRecord]: ...


@runtime_checkable
class Engine(Protocol):
    """Algorithm-specific state machine. Runner is algorithm-agnostic."""
    def ask(self, n: int) -> list[Candidate]: ...
    def tell(self, evaluated: list[Candidate]) -> None: ...
    def metrics(self) -> dict[str, Any]: ...