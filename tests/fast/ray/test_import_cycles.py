from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

CYCLE_PRONE_MODULES: tuple[str, str] = (
    "miles.ray.specs.inference",
    "miles.ray.rollout.router_manager",
)


@pytest.mark.parametrize(
    "module_names",
    [
        CYCLE_PRONE_MODULES,
        tuple(reversed(CYCLE_PRONE_MODULES)),
    ],
    ids=["specs_first", "router_manager_first"],
)
def test_cycle_prone_modules_import_in_either_order(module_names: tuple[str, str]) -> None:
    """A fresh interpreter can import the spec and router-manager modules in either order."""
    program: str = "\n".join(f"import {module_name}" for module_name in module_names)
    env: dict[str, str] = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))

    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"importing {module_names} failed:\n{result.stderr}"
