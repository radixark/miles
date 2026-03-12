"""Verify that key packages carry module docstrings explaining their role."""
from __future__ import annotations

import miles.utils.ft.agents.diagnostics as agents_diag
import miles.utils.ft.controller.diagnostics as controller_diag


class TestDiagnosticsPackageDocstrings:
    """The agents/ and controller/ diagnostics packages serve different roles
    (node-side executors vs cluster-side orchestrators) and their __init__.py
    docstrings should make that distinction clear.
    """

    def test_agents_diagnostics_has_docstring(self) -> None:
        assert agents_diag.__doc__ is not None, (
            "agents/diagnostics/__init__.py should have a docstring"
        )
        assert "node" in agents_diag.__doc__.lower()

    def test_controller_diagnostics_has_docstring(self) -> None:
        assert controller_diag.__doc__ is not None, (
            "controller/diagnostics/__init__.py should have a docstring"
        )
        assert "cluster" in controller_diag.__doc__.lower()
