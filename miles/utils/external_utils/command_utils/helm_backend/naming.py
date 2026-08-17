from __future__ import annotations

import random
import time
from pathlib import Path

from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME, component_name

ORCHESTRATOR_COMPONENT = "orchestrator"

_UNINSTALL_COMPONENT = "uninstall"
_UNINSTALL_MANIFEST_COMPONENT = "uninstall-manifest"

_RUNS_DIR_NAME = "miles-runs"
_STATE_DIR_NAME = "state"
_VALUES_DIR_NAME = "values"
_RECORDS_DIR_NAME = "launches"
_STATE_FILE_GLOB = "orchestrator-*.state"


class RunNames:
    @staticmethod
    def release(*, run_id: str) -> str:
        return f"{CHART_NAME}-{run_id}"

    @staticmethod
    def service_fqdn(*, name: str, namespace: str) -> str:
        return f"{name}.{namespace}.svc.cluster.local"

    @staticmethod
    def orchestrator_object(*, release: str) -> str:
        return component_name(release, ORCHESTRATOR_COMPONENT)

    @staticmethod
    def orchestrator_host(*, release: str, namespace: str) -> str:
        return RunNames.service_fqdn(name=RunNames.orchestrator_object(release=release), namespace=namespace)

    @staticmethod
    def uninstall_job(*, release: str) -> str:
        return component_name(release, _UNINSTALL_COMPONENT)

    @staticmethod
    def uninstall_manifest(*, release: str) -> str:
        return component_name(release, _UNINSTALL_MANIFEST_COMPONENT)


class RunFiles:
    @staticmethod
    def run_dir(*, shared_root: str | Path, run_id: str) -> Path:
        return Path(shared_root) / _RUNS_DIR_NAME / run_id

    @staticmethod
    def new_values_file(*, run_directory: str | Path) -> Path:
        return Path(run_directory) / _VALUES_DIR_NAME / f"values-{_new_launch_token()}.yaml"

    @staticmethod
    def new_state_file(*, run_directory: str | Path) -> Path:
        return _orchestrator_state_path(run_directory, _new_launch_token())

    @staticmethod
    def new_record_file(*, run_directory: str | Path) -> Path:
        return Path(run_directory) / _RECORDS_DIR_NAME / f"launch-{_new_launch_token()}.json"

    @staticmethod
    def latest_state_file(*, run_directory: str | Path) -> Path | None:
        """The newest launch's file, by the launch token in its name; see _new_launch_token."""
        written = sorted((Path(run_directory) / _STATE_DIR_NAME).glob(_STATE_FILE_GLOB))
        return written[-1] if written else None


def _orchestrator_state_path(run_directory: str | Path, launch_token: str) -> Path:
    return Path(run_directory) / _STATE_DIR_NAME / f"orchestrator-{launch_token}.state"


def _new_launch_token() -> str:
    return f"{time.strftime('%y%m%d-%H%M%S')}-{random.Random().randint(0, 999999):06d}"
