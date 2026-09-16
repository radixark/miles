from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
    module_aliases, direct_names = _ray_init_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "init":
            if isinstance(func.value, ast.Name) and func.value.id in module_aliases:
                return True
        if isinstance(func, ast.Name) and func.id in direct_names:
            return True
    return False


class TestCallsRayInit:
    @pytest.mark.parametrize(
        "source",
        [
            "import ray\nray.init()\n",
            "import ray as r\nr.init()\n",
            "from ray import init\ninit()\n",
            "from ray import init as ray_init\nray_init()\n",
            "import ray.util\nray.init()\n",
        ],
    )
    def test_every_way_of_reaching_ray_init_is_detected(self, source: str, tmp_path: Path) -> None:
        """A second cluster owner is just as real when it imports ray under another name."""
        path = tmp_path / "candidate.py"
        path.write_text(source)

        assert _calls_ray_init(path) is True

    @pytest.mark.parametrize(
        "source",
        [
            "import ray\nray.shutdown()\n",
            "from ray import shutdown\nshutdown()\n",
            "def init():\n    pass\n\ninit()\n",
            "import numpy as ray_like\nray_like.init()\n",
        ],
    )
    def test_unrelated_init_calls_are_not_flagged(self, source: str, tmp_path: Path) -> None:
        """Flagging a local init() or another library's would make the guard unusable."""
        path = tmp_path / "candidate.py"
        path.write_text(source)

        assert _calls_ray_init(path) is False


def _ray_init_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Names bound to the ray module and to ray.init itself, so aliased imports still count."""
    module_aliases = {"ray"}
    direct_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ray" or alias.name.startswith("ray."):
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "ray":
            for alias in node.names:
                if alias.name == "init":
                    direct_names.add(alias.asname or alias.name)
    return module_aliases, direct_names
