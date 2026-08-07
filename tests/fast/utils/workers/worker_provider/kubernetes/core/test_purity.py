from __future__ import annotations

import ast
from pathlib import Path

import pytest

from miles.utils.workers.worker_provider.kubernetes import core

FORBIDDEN_IMPORTS = (
    "asyncio",
    "kubernetes_asyncio",
    "miles.utils.workers.worker_provider.kubernetes.core.provider",
    "miles.utils.workers.reconcile.k8s_reflector",
    "miles.utils.workers.reconcile.loop",
)
VIEW_MODULE_NAMES = ("cell_view.py", "pod_view.py")


def view_module_paths() -> list[Path]:
    return [Path(core.__file__).parent / name for name in VIEW_MODULE_NAMES]


def imported_module_names(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _guards_type_checking(node):
            node.body = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.append(node.module)
    return names


def _guards_type_checking(node: ast.If) -> bool:
    return isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"


class TestViewsPurity:
    @pytest.mark.parametrize("source_path", view_module_paths(), ids=lambda path: path.name)
    def test_a_view_module_imports_nothing_that_talks_to_a_cluster(self, source_path: Path) -> None:
        """Views must stay pure projections; an I/O import that runs should fail loudly, not drift silently."""
        offenders = [
            name
            for name in imported_module_names(source_path)
            for forbidden in FORBIDDEN_IMPORTS
            if name == forbidden or name.startswith(f"{forbidden}.")
        ]

        assert not offenders, f"{source_path.name} imports {offenders}, which reaches past a pure projection"

    def test_the_scan_sees_every_view_module(self) -> None:
        """A renamed view module would make the purity check silently scan nothing."""
        assert all(path.exists() for path in view_module_paths())
