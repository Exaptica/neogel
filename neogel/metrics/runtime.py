from __future__ import annotations
from typing import Any

def evals_per_sec(*, evals_total: int, elapsed_sec: float, **_) -> dict[str, Any]:
    if elapsed_sec <= 0:
        return {"evals_per_sec": None}
    return {"evals_per_sec": float(evals_total / elapsed_sec)}