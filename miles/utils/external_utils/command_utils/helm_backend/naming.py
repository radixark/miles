from __future__ import annotations

import random
import time
from pathlib import Path

from pydantic import model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME, component_name

ORCHESTRATOR_COMPONENT = "orchestrator"
PLATFORM_READER_COMPONENT = "platform-reader"

_HELM_RELEASE_NAME_MAX = 53
_LONGEST_COMPONENT_VALUE = max((component.value for component in DeployComponent), key=len)
_LONGEST_COMPONENT_SUFFIX = len(f"-{_LONGEST_COMPONENT_VALUE}")
RUN_ID_MAX_LENGTH = _HELM_RELEASE_NAME_MAX - len(f"{CHART_NAME}-") - _LONGEST_COMPONENT_SUFFIX
_COMPONENT_VALUES = frozenset(component.value for component in DeployComponent)

_UNINSTALL_COMPONENT = "uninstall"
_UNINSTALL_MANIFEST_COMPONENT = "uninstall-manifest"

_RUNS_DIR_NAME = "miles-runs"
_STATE_DIR_NAME = "state"
_VALUES_DIR_NAME = "values"
_RECORDS_DIR_NAME = "launches"
_STATE_FILE_GLOB = "orchestrator-*.state"


class ReleaseName(FrozenStrictBaseModel):
    run_id: str
    deploy_component: DeployComponent
    deploy_instance_id: str | None

    @model_validator(mode="after")
    def _fits_a_helm_release(self) -> ReleaseName:
        assert (
            len(self.run_id) <= RUN_ID_MAX_LENGTH
        ), f"run_id {self.run_id!r} is {len(self.run_id)} characters, at most {RUN_ID_MAX_LENGTH}"
        if self.deploy_instance_id is not None:
            assert self.deploy_instance_id, "deploy_instance_id is empty; a component deployed once carries None"
            budget = _deploy_instance_id_budget(run_id=self.run_id)
            assert len(self.deploy_instance_id) <= budget, (
                f"deploy_instance_id {self.deploy_instance_id!r} is {len(self.deploy_instance_id)} characters, and "
                f"run_id {self.run_id!r} leaves {budget} for it in the release name"
            )
            intersected = sorted(_COMPONENT_VALUES.intersection(self.deploy_instance_id.split("-")))
            assert not intersected, (
                f"--deploy-instance-id {self.deploy_instance_id!r} carries the component name(s) {intersected}, "
                f"which a release name could not be parsed back apart on"
            )
        assert (
            len(name := self.serialize()) <= _HELM_RELEASE_NAME_MAX
        ), f"release {name!r} is {len(name)} characters, at most {_HELM_RELEASE_NAME_MAX}"
        return self

    def serialize(self) -> str:
        parts = [CHART_NAME, self.run_id, self.deploy_component.value]
        if self.deploy_instance_id is not None:
            parts.append(self.deploy_instance_id)
        return "-".join(parts)

    @classmethod
    def parse(cls, release: str) -> ReleaseName | None:
        if not release.startswith(f"{CHART_NAME}-"):
            return None

        tokens = release.removeprefix(f"{CHART_NAME}-").split("-")
        index = max((i for i, token in enumerate(tokens) if token in _COMPONENT_VALUES), default=0)
        if index == 0:
            return None

        return cls(
            run_id="-".join(tokens[:index]),
            deploy_component=DeployComponent(tokens[index]),
            deploy_instance_id="-".join(tokens[index + 1 :]) or None,
        )


def _deploy_instance_id_budget(*, run_id: str) -> int:
    return _HELM_RELEASE_NAME_MAX - len(f"{CHART_NAME}-{run_id}-{_LONGEST_COMPONENT_VALUE}-")


class RunNames:
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
