# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import abc
import random

import requests
from tests.e2e.ft.conftest_ft.fault_injection.pod_manipulation import (
    delete_one_pod_of_cell,
    list_pod_names_of_cell,
    sigkill_process_patterns_in_pod,
)

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.types import ClusterBackend, DeployComponent

FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]
RAY_ROLLOUT_ENGINE_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL]

DELETE_POD_FORM_NAME: str = "delete_pod"
EXEC_SIGKILL_FORM_NAME: str = "exec_sigkill"
ENGINE_CONTAINER_NAME: str = "engine"
SGLANG_PROCESS_PATTERN: str = "sglang::"

ACTOR_CELL_TYPE: str = "actor"
ROLLOUT_CELL_TYPE: str = "rollout"


class BaseFaultForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def harms_the_cell(self) -> bool:
        return True

    @property
    def serialized_against_weight_updates(self) -> bool:
        return False

    @abc.abstractmethod
    def inject(self, cell: dict, rng: random.Random) -> None: ...


class InjectFaultForm(BaseFaultForm):
    def __init__(
        self, *, base_url: str, failure_mode: FailureMode, serialized_against_weight_updates: bool = False
    ) -> None:
        self._base_url = base_url
        self._failure_mode = failure_mode
        self._serialized_against_weight_updates = serialized_against_weight_updates

    @property
    def name(self) -> str:
        return f"inject_fault:{self._failure_mode.value}"

    @property
    def serialized_against_weight_updates(self) -> bool:
        return self._serialized_against_weight_updates

    def inject(self, cell: dict, rng: random.Random) -> None:
        resp = requests.post(
            f"{self._base_url}/api/v1/cells/{cell['metadata']['name']}/inject-fault",
            json={"mode": self._failure_mode.value, "sub_index": 0},
            timeout=5,
        )
        resp.raise_for_status()


class DeletePodFaultForm(BaseFaultForm):
    def __init__(self, *, namespace: str, run_id: str) -> None:
        assert namespace, "Deleting a cell's pod needs the namespace the run was installed into"
        assert run_id, "Deleting a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = ReleaseName(
            run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
        ).serialize()

    @property
    def name(self) -> str:
        return DELETE_POD_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        delete_one_pod_of_cell(
            namespace=self._namespace, release=self._release, cell_id=cell["metadata"]["name"], rng=rng
        )


class ExecSigkillFaultForm(BaseFaultForm):
    def __init__(self, *, namespace: str, run_id: str, container: str, process_pattern: str) -> None:
        assert namespace, "Crashing a process inside a cell's pod needs the namespace the run was installed into"
        assert run_id, "Crashing a process inside a cell's pod needs the run_id naming the release that owns it"

        self._namespace = namespace
        self._release = ReleaseName(
            run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
        ).serialize()
        self._container = container
        self._process_pattern = process_pattern

    @property
    def name(self) -> str:
        return EXEC_SIGKILL_FORM_NAME

    def inject(self, cell: dict, rng: random.Random) -> None:
        cell_id = cell["metadata"]["name"]
        pod_names = list_pod_names_of_cell(namespace=self._namespace, release=self._release, cell_id=cell_id)
        assert pod_names, f"Release {self._release} has no pod of cell {cell_id} in {self._namespace} to crash"

        sigkill_process_patterns_in_pod(
            namespace=self._namespace,
            pod_name=rng.choice(pod_names),
            container=self._container,
            process_pattern=self._process_pattern,
        )


CellFaultForms = dict[str, list[BaseFaultForm]]


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    actor_kill_forms = _inject_fault_forms(base_url=base_url, failure_modes=FAILURE_MODES)

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_kill_forms,
                ROLLOUT_CELL_TYPE: _inject_fault_forms(
                    base_url=base_url,
                    failure_modes=RAY_ROLLOUT_ENGINE_FAILURE_MODES,
                    serialized_against_weight_updates=True,
                ),
            }
        case ClusterBackend.KUBERNETES:
            delete_pod_form = DeletePodFaultForm(namespace=config.namespace, run_id=config.run_id)
            exec_sigkill_form = ExecSigkillFaultForm(
                namespace=config.namespace,
                run_id=config.run_id,
                container=ENGINE_CONTAINER_NAME,
                process_pattern=SGLANG_PROCESS_PATTERN,
            )
            return {
                ACTOR_CELL_TYPE: [*actor_kill_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [exec_sigkill_form, delete_pod_form],
            }


def _inject_fault_forms(
    *, base_url: str, failure_modes: list[FailureMode], serialized_against_weight_updates: bool = False
) -> list[BaseFaultForm]:
    return [
        InjectFaultForm(
            base_url=base_url,
            failure_mode=failure_mode,
            serialized_against_weight_updates=serialized_against_weight_updates,
        )
        for failure_mode in failure_modes
    ]


CELL_TYPE_OF_FT_COMPONENT: dict[str, str] = {"train": ACTOR_CELL_TYPE, "rollout": ROLLOUT_CELL_TYPE}


def compute_mean_interval_seconds_of_cell_type(
    ft_components: tuple[str, ...], *, trainer_crash_interval_seconds: float, rollout_crash_interval_seconds: float
) -> dict[str, float]:
    interval_seconds_of_component: dict[str, float] = {
        "train": trainer_crash_interval_seconds,
        "rollout": rollout_crash_interval_seconds,
    }

    return {
        CELL_TYPE_OF_FT_COMPONENT[component]: interval_seconds_of_component[component] for component in ft_components
    }
