from __future__ import annotations

import importlib
from typing import Any


def import_by_path(path: str) -> Any:
    """Import 'module.submodule:attr' and return attr."""
    if ":" not in path:
        raise ValueError(f"Invalid import path '{path}'. Expected 'module:attr'.")
    mod_name, attr = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise AttributeError(f"Module '{mod_name}' has no attribute '{attr}'.") from e