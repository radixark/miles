import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def modules_imported_by(module: str) -> set[str]:
    script = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


def run_with_blocked_imports(*, script: str, blocked_prefixes: tuple[str, ...]) -> None:
    prelude = f"""
import importlib.abc
import sys

blocked = {blocked_prefixes!r}

class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("Forbidden eager import: " + fullname)
        return None

assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
sys.meta_path.insert(0, BlockImports())
"""
    completed = subprocess.run(
        [sys.executable, "-c", prelude + script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
