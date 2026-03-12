"""Verify that shared constants and data types live in utils/, not in layer-specific packages.

The agents/ and controller/ layers used to have cross-layer imports for shared
constants (metric names, GPU XID constants) and data types (DiagnosticResult).
These are now in utils/ so that both layers import from the common base instead
of reaching into each other.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_FT_ROOT = Path(__file__).resolve().parents[4] / "miles" / "utils" / "ft"


def _collect_imports(module_path: Path) -> list[str]:
    """Return all absolute import strings from a Python file."""
    source = module_path.read_text()
    tree = ast.parse(source)
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


class TestMetricNamesLivesInUtils:
    def test_metric_names_importable_from_utils(self) -> None:
        mod = importlib.import_module("miles.utils.ft.utils.metric_names")
        assert hasattr(mod, "GPU_AVAILABLE")
        assert hasattr(mod, "ROLLOUT_CELL_ALIVE")

    def test_agents_collectors_do_not_import_metric_names_from_controller(self) -> None:
        """Agents collectors used to import metric_names from controller.metrics,
        creating an agents→controller reverse dependency.
        """
        collectors_dir = _FT_ROOT / "agents" / "collectors"
        for py_file in collectors_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            imports = _collect_imports(py_file)
            for imp in imports:
                assert "controller.metrics.metric_names" not in imp, (
                    f"{py_file.name} still imports metric_names from controller"
                )


class TestGpuConstantsLiveInUtils:
    def test_gpu_constants_importable_from_utils(self) -> None:
        mod = importlib.import_module("miles.utils.ft.utils.gpu_constants")
        assert hasattr(mod, "NON_AUTO_RECOVERABLE_XIDS")

    def test_kmsg_collector_does_not_import_xid_catalog_from_controller(self) -> None:
        """kmsg.py used to import NON_AUTO_RECOVERABLE_XIDS directly from
        controller.detectors.checks.gpu.xid_catalog.info, crossing layers.
        """
        kmsg_path = _FT_ROOT / "agents" / "collectors" / "kmsg.py"
        imports = _collect_imports(kmsg_path)
        for imp in imports:
            assert "xid_catalog" not in imp, (
                "kmsg.py still imports from xid_catalog directly"
            )


class TestDiagnosticTypesLiveInUtils:
    def test_diagnostic_types_importable_from_utils(self) -> None:
        mod = importlib.import_module("miles.utils.ft.utils.diagnostic_types")
        assert hasattr(mod, "DiagnosticResult")
        assert hasattr(mod, "DiagnosticPipelineResult")
        assert hasattr(mod, "UnknownDiagnosticError")

    def test_controller_types_does_not_import_from_agents(self) -> None:
        """controller/types.py used to import DiagnosticPipelineResult from
        agents/types.py, creating a controller→agents dependency.
        """
        types_path = _FT_ROOT / "controller" / "types.py"
        imports = _collect_imports(types_path)
        for imp in imports:
            assert ".agents.types" not in imp, (
                "controller/types.py still imports from agents.types"
            )

    def test_adapters_types_does_not_import_from_agents(self) -> None:
        """adapters/types.py used to import DiagnosticResult from agents/types.py,
        creating an adapters→agents dependency.
        """
        types_path = _FT_ROOT / "adapters" / "types.py"
        imports = _collect_imports(types_path)
        for imp in imports:
            assert ".agents.types" not in imp, (
                "adapters/types.py still imports from agents.types"
            )
