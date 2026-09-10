import asyncio
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import typer
import yaml
from tests.e2e.deploy.conftest_deploy.hot_restart.deployment_target import validate_deployment_target
from tests.e2e.deploy.conftest_deploy.hot_restart.guard_manifest import GUARDED_WORKLOAD_KINDS
from tests.utils.soak.recipes.gsm8k import launch_gsm8k
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec
from tests.utils.soak.state import SoakDeploymentTarget

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.uninstall_lock import uninstall_lock
from miles.utils.workers.cell_operations.base import StaleFaultTargetError
from miles.utils.workers.serving.utils import override_env


class HotRestartLaunchSpec(Gsm8kLaunchSpec):
    target: SoakDeploymentTarget
    guard_directory: Path


def launch_guarded(spec: HotRestartLaunchSpec) -> None:
    target = spec.target
    if target.state_file is None:
        raise StaleFaultTargetError("Hot restart requires the observed orchestrator state file")
    original_defuse = entrypoint._defuse_previous_generation
    original_delete_job = Kubectl.delete_job
    original_get_manifest = Helm.get_manifest

    def get_manifest(release: str, namespace: str) -> Manifest | None:
        manifest = original_get_manifest(release, namespace)
        if manifest is None or (release, namespace) != (target.release, target.namespace):
            return manifest
        return _without_guard_preconditions(manifest)

    def defuse(release: str, *, namespace: str, superseded_state_file: Path | None, state_file: Path | None) -> None:
        if (release, namespace, superseded_state_file) != (target.release, target.namespace, target.state_file):
            raise StaleFaultTargetError("The launcher no longer carries the observed orchestrator generation")
        with uninstall_lock(target.state_file):
            asyncio.run(validate_deployment_target(target))
            original_defuse(
                release, namespace=namespace, superseded_state_file=superseded_state_file, state_file=state_file
            )

    def delete_job(name: str, *, namespace: str, check: bool = False) -> None:
        if name == RunNames.uninstall_job(release=target.release) and namespace == target.namespace:
            _delete_observed_job(name=name, namespace=namespace, expected_uid=target.uninstall_job_uid)
        else:
            original_delete_job(name, namespace=namespace, check=check)

    def upgrade(
        *, release: str, namespace: str, chart: str | Path, values_files: list[str | Path], ci_run: bool
    ) -> None:
        if (release, namespace) != (target.release, target.namespace):
            raise StaleFaultTargetError("The upgrade targets another release")
        plugin_root = _write_guard_plugin(target=target, directory=spec.guard_directory)
        with override_env({"HELM_PLUGINS": str(plugin_root)}):
            run_process(
                [
                    *Helm.upgrade_command(release, namespace, chart, values_files, ci_run=ci_run),
                    "--post-renderer",
                    "miles-soak-guard",
                    "--server-side=true",
                ],
                capture_output=False,
                check=True,
                timeout=300,
            )

    with patch.object(entrypoint, "_defuse_previous_generation", defuse), patch.object(
        Kubectl, "delete_job", delete_job
    ), patch.object(Helm, "upgrade", upgrade), patch.object(Helm, "get_manifest", get_manifest):
        launch_gsm8k(config=spec.config, train_args=spec.train_args, fully_async=spec.fully_async)


def _without_guard_preconditions(manifest: Manifest) -> Manifest:
    documents = [document.body for document in manifest.objects]
    for document in documents:
        if document["kind"] not in GUARDED_WORKLOAD_KINDS:
            continue
        document["metadata"].pop("uid", None)
        document["metadata"].pop("resourceVersion", None)
    return Manifest(namespace=manifest.namespace, objects=documents)


def _write_guard_plugin(*, target: SoakDeploymentTarget, directory: Path) -> Path:
    plugin = directory / "miles-soak-guard"
    plugin.mkdir(parents=True, exist_ok=False)
    target_path = plugin / "target.json"
    target_path.write_text(target.model_dump_json(indent=2))
    script = plugin / "render.sh"
    script.write_text(
        "#!/bin/sh\nexec "
        + shlex.join(
            [
                sys.executable,
                "-m",
                "tests.e2e.deploy.conftest_deploy.hot_restart.guard_manifest",
                str(target_path),
            ]
        )
        + "\n"
    )
    script.chmod(0o700)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "type": "postrenderer/v1",
                "name": "miles-soak-guard",
                "version": "0.1.0",
                "runtime": "subprocess",
                "runtimeConfig": {"platformCommand": [{"command": str(script)}]},
            },
            sort_keys=False,
        )
    )
    return directory


def _delete_observed_job(*, name: str, namespace: str, expected_uid: str | None) -> None:
    current = run_process(
        ["kubectl", "get", "job", name, "--namespace", namespace, "--output", "json", "--ignore-not-found"],
        capture_output=True,
        check=True,
        timeout=60,
    )
    current_uid = json.loads(current.stdout)["metadata"]["uid"] if current.stdout.strip() else None
    if current_uid != expected_uid:
        raise StaleFaultTargetError("The uninstall job changed since observation")
    if expected_uid is None:
        return
    run_process(
        [
            "kubectl",
            "delete",
            "--raw",
            f"/apis/batch/v1/namespaces/{quote(namespace, safe='')}/jobs/{quote(name, safe='')}",
            "--filename",
            "-",
        ],
        input=json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": expected_uid},
                "propagationPolicy": "Foreground",
            }
        ),
        capture_output=True,
        check=True,
        timeout=60,
    )
    run_process(
        ["kubectl", "wait", "--for=delete", f"job/{name}", "--namespace", namespace, "--timeout=60s"],
        capture_output=True,
        check=True,
        timeout=65,
    )


app = typer.Typer()


@app.command()
def main() -> None:
    launch_guarded(HotRestartLaunchSpec.model_validate_json(sys.stdin.read()))


if __name__ == "__main__":
    app()
