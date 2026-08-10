from __future__ import annotations

import ast
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOTS_ROOT = _TESTS_ROOT / "snapshots"


def test_only_the_session_fixture_starts_the_ray_cluster():
    """A CPU suite runs many files in one pytest process, so whichever file calls ray.init
    first fixes the cluster's resources for every later file. A second initializer there
    silently produces a cluster without the logical GPUs the placement-group tests need."""
    offenders = sorted(str(path.relative_to(_TESTS_ROOT)) for path in _test_files_calling_ray_init())

    assert offenders == ["conftest.py"], (
        f"{offenders} call ray.init directly; depend on the session-scoped ray_local_mode "
        f"fixture in tests/conftest.py instead so the cluster has one owner"
    )


def _test_files_calling_ray_init() -> list[Path]:
    # Snapshot fixtures include directories named like modules, and the launch scripts
    # recorded under them are rendered output rather than code this suite runs.
    candidates = (path for path in _TESTS_ROOT.rglob("*.py") if path.is_file())
    return [path for path in candidates if _SNAPSHOTS_ROOT not in path.parents and _calls_ray_init(path)]


def _calls_ray_init(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "init":
            if isinstance(func.value, ast.Name) and func.value.id == "ray":
                return True
    return False
