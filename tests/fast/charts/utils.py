import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CHARTS_DIR = REPO_ROOT / "charts"
CHART_DIR = CHARTS_DIR / "miles-workbench"
RUN_CHART_DIR = CHARTS_DIR / "miles-run"
SHARED_INFRA_SCHEMA_PATH = CHARTS_DIR / "shared-infra.schema.json"
WORKBENCH_PACKAGE = "miles.utils.external_utils.miles_workbench"

RELEASE_NAME = "miles-workbench-myuser"
UNINSTALLER_SERVICE_ACCOUNT = "miles-uninstaller"
RUN_RELEASE_NAME = "myrun"
RUN_ORCHESTRATOR_NAME = f"{RUN_RELEASE_NAME}-miles-run-orchestrator"
RUN_UNINSTALL_JOB_NAME = f"{RUN_RELEASE_NAME}-miles-run-uninstall"
RUN_UNINSTALL_MANIFEST_NAME = f"{RUN_RELEASE_NAME}-miles-run-uninstall-manifest"
RUN_ID = "260101-000000-000"
RUN_STATE_FILE = f"/cluster-storage/miles_data/miles-runs/{RUN_ID}/state/orchestrator.state"
DEFAULT_ORCHESTRATOR_COMMAND = ["python", "train.py"]
OBJECT_NAME = RELEASE_NAME
NAMESPACE = "myns"
LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"

requires_helm = pytest.mark.skipif(
    shutil.which("helm") is None and not os.environ.get("CI"),
    reason="helm is required to render charts; CI installs it, so a CI run fails instead of skipping",
)


def run_helm_template(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "helm",
            "template",
            RELEASE_NAME,
            str(CHART_DIR),
            "-n",
            NAMESPACE,
            "--set",
            f"objectName={OBJECT_NAME}",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def run_helm_lint(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["helm", "lint", str(CHART_DIR), "--set", f"objectName={OBJECT_NAME}", *args], capture_output=True, text=True
    )


def render(*args: str) -> list[dict[str, Any]]:
    result = run_helm_template(*args)
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document is not None]


def render_error(*args: str) -> str:
    result = run_helm_template(*args)
    assert result.returncode != 0, result.stdout
    return result.stderr


def schema_error_mentions(error: str, *, path: tuple[str, ...]) -> bool:
    return "/" + "/".join(path) in error or ".".join(path) in error


def run_helm_template_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "helm",
            "template",
            RUN_RELEASE_NAME,
            str(RUN_CHART_DIR),
            "-n",
            NAMESPACE,
            "--set",
            f"run.id={RUN_ID}",
            "--set",
            f"run.stateFile={RUN_STATE_FILE}",
            "--set",
            f"run.objectNames.orchestrator={RUN_ORCHESTRATOR_NAME}",
            "--set",
            f"run.objectNames.mooncakeMaster={RUN_RELEASE_NAME}-miles-run-mooncake-master",
            "--set",
            f"run.objectNames.colocatePairing={RUN_RELEASE_NAME}-miles-run-colocate-pairing",
            "--set",
            f"run.objectNames.uninstall={RUN_UNINSTALL_JOB_NAME}",
            "--set",
            f"run.objectNames.uninstallManifest={RUN_UNINSTALL_MANIFEST_NAME}",
            "--set-json",
            f"run.orchestrator.command={json.dumps(DEFAULT_ORCHESTRATOR_COMMAND)}",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def with_object_names(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**entry, "objectName": f"{RUN_RELEASE_NAME}-miles-run-{entry['name']}"} for entry in entries]


def render_run(*args: str) -> list[dict[str, Any]]:
    result = run_helm_template_run(*args)
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document is not None]


def render_run_error(*args: str) -> str:
    result = run_helm_template_run(*args)
    assert result.returncode != 0, result.stdout
    return result.stderr


def named_object(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    matched = [obj for obj in objects_of_kind(objects, kind) if obj["metadata"]["name"] == name]
    assert len(matched) == 1, f"expected one {kind}/{name}, got {[obj['metadata']['name'] for obj in matched]}"
    return matched[0]


def pod_spec_of(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    return named_object(objects, kind, name)["spec"]["template"]["spec"]


def sole_container_of(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    containers = pod_spec_of(objects, kind, name)["containers"]
    assert len(containers) == 1
    return containers[0]


def run_workbench(*args: str, interpreter: str | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    command = [interpreter or sys.executable, "-m", WORKBENCH_PACKAGE, *args]
    # rich colours an option name in pieces, so a coloured '--namespace' contains no such
    # substring; a runner that reports colour support would hide every message assertion here
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", **kwargs.pop("env", {})}
    return subprocess.run(command, cwd=REPO_ROOT, text=True, timeout=60, env=env, **kwargs)


def merged_rules(*rules: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = {}
    for rule in rules:
        for resource, verbs in rule.items():
            merged.setdefault(resource, set()).update(verbs)
    return {resource: tuple(sorted(verbs)) for resource, verbs in merged.items()}


def can_i_queries(rules: dict[str, tuple[str, ...]]) -> set[str]:
    queries = set()
    for resource, verbs in rules.items():
        target, _, subresource = resource.partition("/")
        for verb in verbs:
            queries.add(f"{verb} {target} --subresource={subresource}" if subresource else f"{verb} {target}")
    return queries


def resolved_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text())
    return _resolve_refs(schema, schema.get("definitions", {}))


def _resolve_refs(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_resolve_refs(entry, definitions) for entry in node]
    if not isinstance(node, dict):
        return node
    if (reference := node.get("$ref")) is not None:
        return _resolve_refs(definitions[reference.rsplit("/", maxsplit=1)[1]], definitions)
    return {key: _resolve_refs(value, definitions) for key, value in node.items() if key != "definitions"}


def chart_directories() -> list[Path]:
    chart_dirs = sorted(
        path.parent
        for path in SHARED_INFRA_SCHEMA_PATH.parent.glob("*/Chart.yaml")
        if "type: library" not in path.read_text()
    )
    assert chart_dirs, "no application chart found under charts/"
    return chart_dirs


def library_chart_directories() -> list[Path]:
    return sorted(
        path.parent
        for path in SHARED_INFRA_SCHEMA_PATH.parent.glob("*/Chart.yaml")
        if "type: library" in path.read_text()
    )


def objects_of_kind(objects: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [obj for obj in objects if obj["kind"] == kind]


def single_object_of_kind(objects: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matched = objects_of_kind(objects, kind)
    assert len(matched) == 1, f"expected exactly one {kind}, got {[obj['metadata']['name'] for obj in matched]}"
    return matched[0]


def pod_spec(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]


def container(objects: list[dict[str, Any]]) -> dict[str, Any]:
    containers = pod_spec(objects)["containers"]
    assert len(containers) == 1
    return containers[0]
