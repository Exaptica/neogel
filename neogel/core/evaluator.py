from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Sequence

from .types import EvalRecord, Evaluator, EvaluateFn


class SerialEvaluator(Evaluator):
    def map(self, fn: EvaluateFn, items: Sequence[Any]) -> list[EvalRecord]:
        return [fn(x) for x in items]


class ProcessPoolEvaluator(Evaluator):
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def map(self, fn: Callable[[Any], EvalRecord], items: Sequence[Any]) -> list[EvalRecord]:
        # NOTE: fn must be picklable.
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(fn, items))