from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.miles_workbench import render as render_module
from miles.utils.external_utils.miles_workbench.options import InstallArgs


class TestRenderChartFrom:
    def test_a_failed_dependency_build_is_returned_without_rendering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed dependency build is returned without attempting to render the chart."""
        failure = subprocess.CompletedProcess(
            args=["helm", "dependency", "build", "/chart"], returncode=9, stdout="build output", stderr="build error"
        )

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            if arguments[0] == "template":
                raise AssertionError("rendering must not occur after dependency-build failure")
            return failure

        monkeypatch.setattr(render_module.Helm, "run_raw", staticmethod(run_raw))
        args = InstallArgs(
            namespace="rl",
            release="workbench",
            rbac=True,
            lws=True,
            dry_run=False,
            values=(),
            overrides=(),
            skip_preflight=False,
            timeout=60,
        )

        result = render_module._render_chart_from(args, chart_dir=Path("/chart"))

        assert result is failure
