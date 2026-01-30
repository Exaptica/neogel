from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from neogel.core.evaluator import ProcessPoolEvaluator, SerialEvaluator
from neogel.core.types import EvalRecord


# IMPORTANT: must be top-level to be picklable by multiprocessing on macOS/Windows.
def eval_with_pid(x: int) -> EvalRecord:
    # Return PID so we can confirm multiple worker processes ran tasks.
    pid = os.getpid()
    # Dummy objective: maximize x
    return EvalRecord(objectives=np.array([float(x)], dtype=float), extras={"pid": pid})


def test_process_pool_evaluator_uses_multiple_processes():
    items = list(range(200))

    # Serial baseline
    serial = SerialEvaluator()
    serial_records = serial.map(eval_with_pid, items)
    serial_pids = {r.extras["pid"] for r in serial_records if r.extras and "pid" in r.extras}

    # In serial, all evaluations should come from the current process PID.
    assert len(serial_pids) == 1

    # Process pool
    pool = ProcessPoolEvaluator(max_workers=4)
    pool_records = pool.map(eval_with_pid, items)
    pool_pids = {r.extras["pid"] for r in pool_records if r.extras and "pid" in r.extras}

    # Confirm we used >1 OS process (workers).
    assert len(pool_pids) > 1

    # Confirm results are correct and consistent
    serial_objs = [float(r.objectives[0]) for r in serial_records]
    pool_objs = [float(r.objectives[0]) for r in pool_records]
    assert serial_objs == pool_objs