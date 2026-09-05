from __future__ import annotations

from pathlib import Path

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    WORKBENCH_OBJECT_NAME_MAX,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import release_prefix

CHART_DIR = Path(__file__).resolve().parents[4] / "charts" / "miles-workbench"
CHART_NAME = "miles-workbench"

DEFAULT_RELEASE = "workbench"

PACKAGE = "miles.utils.external_utils.miles_workbench"
PROGRAM_NAME = f"python -m {PACKAGE}"


def run_release_name(
    run_id: str, deploy_component: DeployComponent = DeployComponent.ALL, deploy_instance_id: str | None = None
) -> str:
    return ReleaseName(
        run_id=run_id, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
    ).serialize()


def object_name(release: str) -> str:
    return release_prefix(release, chart_name=CHART_NAME, budget=WORKBENCH_OBJECT_NAME_MAX)
