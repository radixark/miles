from __future__ import annotations

import ast
from pathlib import Path

import pytest

from miles.utils.workers.worker_provider.kubernetes import core

HELM_PACKAGE = "miles.utils.workers.worker_provider.kubernetes.helm"
CHART_LITERALS = ("radixark.io", "leaderworkerset.sigs.k8s.io", "app.kubernetes.io/instance")
LAUNCH_SIDE_SPEC_FIELDS = ("env_var", "ctor_kwargs", "launch_command")
OBSERVING_MODULE_NAMES = ("provider.py", "pod_view.py", "cell_view.py")


def core_module_paths() -> list[Path]:
    return sorted(Path(core.__file__).parent.rglob("*.py"))


def imported_module_names(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.append(node.module)
    return names


class TestCoreKnowsNoChart:
    @pytest.mark.parametrize("source_path", core_module_paths(), ids=lambda path: path.name)
    def test_a_core_module_never_imports_the_helm_glue(self, source_path: Path) -> None:
        """Core implements kubernetes in general; importing the chart's glue would invert the layering."""
        offenders = [
            name
            for name in imported_module_names(source_path)
            if name == HELM_PACKAGE or name.startswith(f"{HELM_PACKAGE}.")
        ]

        assert not offenders, f"{source_path.name} imports {offenders}, which belongs to the chart above it"

    @pytest.mark.parametrize("source_path", core_module_paths(), ids=lambda path: path.name)
    def test_a_core_module_spells_no_chart_label(self, source_path: Path) -> None:
        """A label key baked into core would make the layer describe one deployment rather than kubernetes."""
        source = source_path.read_text()
        offenders = [literal for literal in CHART_LITERALS if literal in source]

        assert not offenders, f"{source_path.name} spells {offenders}, which only the chart's glue may name"

    def test_the_scan_sees_every_core_module(self) -> None:
        """A glob that matched nothing would make the layering check vacuously green."""
        assert {path.name for path in core_module_paths()} >= {"provider.py", "pod_view.py", "cell_view.py"}


class TestObserversReadOnlyWhatTheyObserve:
    @pytest.mark.parametrize("module_name", OBSERVING_MODULE_NAMES)
    def test_an_observing_module_never_names_a_launch_side_spec_field(self, module_name: str) -> None:
        """The run carries whole specs, so only a test can still stop the reconciler from launching workers."""
        source_path = Path(core.__file__).parent / module_name
        source = source_path.read_text()

        offenders = [field for field in LAUNCH_SIDE_SPEC_FIELDS if field in source]

        assert not offenders, f"{module_name} names {offenders}, which describes how a worker is started"
