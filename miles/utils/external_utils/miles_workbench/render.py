from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.miles_workbench.naming import CHART_DIR
from miles.utils.external_utils.miles_workbench.options import InstallArgs
from miles.utils.external_utils.miles_workbench.preflight.rules import LWS_RESOURCE
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class RbacPlan(FrozenStrictBaseModel):
    creates_role: bool
    granted_rules: dict[str, tuple[str, ...]] = {}

    @property
    def grants_leader_worker_sets(self) -> bool:
        return LWS_RESOURCE in self.granted_rules


def render_chart(args: InstallArgs) -> subprocess.CompletedProcess[str]:
    if not args.dry_run:
        return _render_chart_from(args, chart_dir=CHART_DIR)

    with tempfile.TemporaryDirectory() as scratch:
        charts_copy = Path(scratch) / CHART_DIR.parent.name
        shutil.copytree(CHART_DIR.parent, charts_copy)
        return _render_chart_from(args, chart_dir=charts_copy / CHART_DIR.name)


def _render_chart_from(args: InstallArgs, *, chart_dir: Path) -> subprocess.CompletedProcess[str]:
    build = Helm.run_raw("dependency", "build", str(chart_dir))
    if build.returncode != 0:
        return build
    return Helm.run_raw("template", args.release, str(chart_dir), "-n", args.namespace, *helm_value_overrides(args))


def rbac_plan_of(rendered: str) -> RbacPlan:
    roles = [
        document
        for document in yaml.safe_load_all(rendered)
        if isinstance(document, dict) and document.get("kind") == "Role"
    ]
    granted: dict[str, tuple[str, ...]] = {}
    for role in roles:
        for rule in role.get("rules") or []:
            for resource, verbs in _rule_entries(rule).items():
                granted[resource] = tuple(sorted({*granted.get(resource, ()), *verbs}))
    return RbacPlan(creates_role=bool(roles), granted_rules=granted)


def _rule_entries(rule: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    assert "resourceNames" not in rule, (
        f"The chart grants {rule} on named objects only, but a preflight check asks kubectl about a whole "
        f"resource; qualify the check with the name before shipping a rule like this"
    )
    verbs = tuple(rule["verbs"])
    entries = {}
    for group in rule["apiGroups"]:
        for resource in rule["resources"]:
            name, _, subresource = resource.partition("/")
            key = name if group == "" else f"{name}.{group}"
            entries[f"{key}/{subresource}" if subresource else key] = verbs
    return entries


def helm_value_overrides(args: InstallArgs) -> list[str]:
    overrides: list[str] = []
    if not args.rbac:
        overrides += ["--set", "rbac.create=false"]
    if not args.lws:
        overrides += ["--set", "rbac.leaderWorkerSets=false"]
    for values_file in args.values:
        overrides += ["-f", str(values_file)]
    for override in args.overrides:
        overrides += ["--set", override]
    return overrides
