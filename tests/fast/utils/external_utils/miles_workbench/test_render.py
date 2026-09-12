from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.miles_workbench import render as render_module
from miles.utils.external_utils.miles_workbench.naming import object_name
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.render import rbac_plan_of


def _install_args(*, release: str = "workbench") -> InstallArgs:
    return InstallArgs(
        namespace="rl",
        release=release,
        rbac=True,
        lws=True,
        dry_run=False,
        values=(),
        overrides=(),
        skip_preflight=False,
        timeout=60,
    )


class TestRenderChartFrom:
    def test_the_chart_is_rendered_under_the_object_name_the_install_will_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rendering without it names every Role and ServiceAccount empty, so the preflight can only fail."""
        rendered: list[tuple[str, ...]] = []

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            rendered.append(arguments)
            return subprocess.CompletedProcess(args=list(arguments), returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_module.Helm, "run_raw", staticmethod(run_raw))

        render_module._render_chart_from(_install_args(release="wb"), chart_dir=Path("/chart"))

        assert "--set-string" in rendered[-1]
        assert f"objectName={object_name('wb')}" in rendered[-1]

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

        result = render_module._render_chart_from(_install_args(), chart_dir=Path("/chart"))

        assert result is failure


class TestRbacPlanOf:
    def test_a_named_resource_grant_is_rejected_before_creating_a_whole_resource_plan(self) -> None:
        """A named-object grant is rejected before becoming a whole-resource permission check."""
        rendered = """
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: workbench
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["workbench-token"]
    verbs: ["get"]
"""

        with pytest.raises(AssertionError, match="named objects only"):
            rbac_plan_of(rendered)
