from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _run_neogel(config_path: Path, extra_args: list[str]) -> float:
    """Run neogel CLI and return walltime_sec parsed from stdout."""
    cmd = ["neogel", "run", str(config_path), "--no-rich", "--tag", "pytest"] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"neogel failed.\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # Expect: DONE: {'walltime_sec': 12.34, ...}
    m = re.search(r"walltime_sec'\s*:\s*([0-9]*\.[0-9]+|[0-9]+)", proc.stdout)
    if not m:
        raise AssertionError(f"Could not parse walltime_sec from stdout:\n{proc.stdout}")

    return float(m.group(1))


def test_cli_serial_runs(tmp_path: Path):
    # Run from repo root to ensure configs paths resolve
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "configs" / "ga" / "sleepy_sphere.yaml"

    # Put runs under a temp directory so tests don't pollute repo
    t = _run_neogel(config, ["--runs-root", str(tmp_path), "--seed", "0"])
    assert t > 0.0


def test_cli_parallel_faster_than_serial(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "configs" / "ga" / "sleepy_sphere.yaml"

    # Serial baseline
    t_serial = _run_neogel(config, ["--runs-root", str(tmp_path), "--seed", "0"])

    # Parallel run
    t_parallel = _run_neogel(config, ["--runs-root", str(tmp_path), "--seed", "0", "--workers", "4"])

    # Require a meaningful speedup. With sleep-based evaluation, this should be robust.
    # Use a conservative threshold so it doesn't flake on slower machines.
    assert t_parallel < 0.75 * t_serial, f"Expected parallel faster. serial={t_serial:.3f}s parallel={t_parallel:.3f}s"