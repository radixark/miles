import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from tests.utils.soak.deploy.guard.target_check import assert_workloads_unchanged
from tests.utils.soak.deploy.types import DeploymentTarget
from tests.utils.soak.recipes.gsm8k import Gsm8kLaunchSpec

from miles.utils.external_utils.command_utils.base_backend import LaunchGuard
from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.cell_operations.base import StaleFaultTargetError

GUARDED_UPGRADE_TIMEOUT_SECONDS: float = 300.0


class HotRestartLaunchSpec(Gsm8kLaunchSpec):
    target: DeploymentTarget


@dataclass(frozen=True, kw_only=True)
class HotRestartLaunchGuard(LaunchGuard):
    target: DeploymentTarget

    def __post_init__(self) -> None:
        if self.target.state_file is None:
            raise StaleFaultTargetError("Hot restart requires the observed orchestrator state file")

    def before_defuse(
        self, release: str, *, namespace: str, superseded_state_file: Path | None, state_file: Path | None
    ) -> None:
        target = self.target
        if (release, namespace, superseded_state_file) != (target.release, target.namespace, target.state_file):
            raise StaleFaultTargetError("The launcher no longer carries the observed orchestrator generation")
        assert_workloads_unchanged(target)

    def delete_uninstall_job(self, name: str, *, namespace: str, check: bool = False) -> None:
        target = self.target
        if (name, namespace) != (RunNames.uninstall_job(release=target.release), target.namespace):
            raise StaleFaultTargetError("The launcher deletes an uninstall job of another release")
        _delete_observed_job(name=name, namespace=namespace, expected_uid=target.uninstall_job_uid)

    def upgrade(
        self, *, release: str, namespace: str, chart: str | Path, values_files: list[str | Path], ci_run: bool
    ) -> None:
        if (release, namespace) != (self.target.release, self.target.namespace):
            raise StaleFaultTargetError("The upgrade targets another release")
        Helm.upgrade(
            release=release,
            namespace=namespace,
            chart=chart,
            values_files=values_files,
            ci_run=ci_run,
            timeout=GUARDED_UPGRADE_TIMEOUT_SECONDS,
        )


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
