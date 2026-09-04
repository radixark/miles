"""Machine enforcement of the gateway layering law: core speaks only the
internal language, server never touches miles infrastructure, and runtime.py
is the single miles translator inside the package."""

import ast
import sys
from pathlib import Path

TINKER_DIR = Path(__file__).resolve().parents[3] / "miles" / "tinker"
STDLIB = set(sys.stdlib_module_names)


def _import_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module)
    return roots


def test_core_imports_only_stdlib_and_itself():
    for path in sorted((TINKER_DIR / "core").glob("*.py")):
        for module in _import_roots(path):
            allowed = module.split(".")[0] in STDLIB or module.startswith("miles.tinker.core")
            assert allowed, f"{path.name} imports {module}; core must stay stdlib-only"


def test_server_never_imports_miles_infrastructure():
    for path in sorted((TINKER_DIR / "server").glob("*.py")):
        for module in _import_roots(path):
            assert not module.startswith(
                ("miles.ray", "miles.backends", "miles.utils")
            ), f"{path.name} imports {module}; server may only speak SDK wire and core"


def test_runtime_is_the_only_miles_speaker():
    for path in sorted(TINKER_DIR.rglob("*.py")):
        if path.name == "runtime.py":
            continue
        for module in _import_roots(path):
            assert not module.startswith(
                ("miles.ray", "miles.backends")
            ), f"{path.relative_to(TINKER_DIR)} imports {module}; only runtime.py translates miles"
