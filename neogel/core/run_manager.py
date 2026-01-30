from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_slug(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s).strip("_")


def _git_short_hash(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len(out.strip()) > 0
    except Exception:
        return None


@dataclass(slots=True)
class RunPaths:
    run_dir: Path
    artifacts_dir: Path
    metrics_csv: Path
    meta_json: Path


class RunManager:
    def __init__(self, *, root: str | Path = "runs"):
        self.root = Path(root)

    def create(
        self,
        *,
        experiment: str,
        seed: int | None = None,
        tag: str | None = None,
        cwd_for_git: str | Path | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> RunPaths:
        exp = _safe_slug(experiment)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"seed{seed}" if seed is not None else "seedNA"
        cwd = Path(cwd_for_git) if cwd_for_git is not None else Path.cwd()

        gith = _git_short_hash(cwd) or "nogit"
        dirty = _git_dirty(cwd)
        dirty_str = "dirty" if dirty else "clean" if dirty is not None else "unknown"

        tag_str = f"_{_safe_slug(tag)}" if tag else ""
        run_name = f"{ts}_{seed_str}_{gith}_{dirty_str}{tag_str}"

        run_dir = self.root / exp / run_name
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=False)

        paths = RunPaths(
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            metrics_csv=run_dir / "metrics.csv",
            meta_json=run_dir / "meta.json",
        )

        meta = {
            "experiment": exp,
            "timestamp": ts,
            "seed": seed,
            "tag": tag,
            "git": {"short_hash": gith, "dirty": dirty},
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        }
        if extra_meta:
            meta["extra"] = extra_meta

        with open(paths.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return paths