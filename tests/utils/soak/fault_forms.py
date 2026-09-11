# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import asyncio
import logging
from typing import Literal

import httpx
from tests.utils.soak.action import SoakActionForm, run_command
from tests.utils.soak.pod_manipulation import delete_observed_pod
from tests.utils.soak.process_target import ProcessExitReceipt, ProcessStopReceipt
from tests.utils.soak.state import SoakActionRequest, SoakPodTarget

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.ft_utils.api_server.fault_receipts import FaultDeadlockReceipt, FaultReceipt, FaultStopReceipt
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS
from miles.utils.workers.types import ClusterBackend, DeployComponent

logger = logging.getLogger(__name__)

FAILURE_MODES: list[FailureMode] = [
    FailureMode.SIGKILL,
    FailureMode.EXIT,
    FailureMode.SEGFAULT,
    FailureMode.SIGSTOP,
    FailureMode.DEADLOCK,
]
RAY_ROLLOUT_ENGINE_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.SIGSTOP]

DELETE_POD_FORM_NAME: str = "delete_pod"
EXEC_SIGKILL_FORM_NAME: str = "exec_sigkill"
EXEC_SIGSTOP_FORM_NAME: str = "exec_sigstop"
ENGINE_CONTAINER_NAME: str = "engine"
SGLANG_PROCESS_PATTERN: str = "sglang::"

ACTOR_CELL_TYPE: str = "actor"
ROLLOUT_CELL_TYPE: str = "rollout"


class InjectFaultForm(SoakActionForm):
    def __init__(self, *, base_url: str, failure_mode: FailureMode) -> None:
        self._base_url = base_url
        self._failure_mode = failure_mode

    @property
    def name(self) -> str:
        return f"inject_fault:{self._failure_mode.value}"

    async def execute(self, request: SoakActionRequest) -> dict:
        assert request.form_name == self.name, f"Request {request.request_id} names another form: {request.form_name}"
        assert isinstance(request.target, dict), "Fault injection requires a cell target"
        assert request.fault_target is not None, "Fault injection requires an observed process identity"
        assert request.fault_target.cell_id == request.target["metadata"]["name"]
        assert request.fault_target.workers_hash == request.target["status"]["workers_hash"]
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/v1/cells/{request.target['metadata']['name']}/inject-fault",
                    json={
                        "mode": self._failure_mode.value,
                        "sub_index": request.fault_target.sub_index,
                        "expected_target": request.fault_target.model_dump(mode="json"),
                        "request_id": request.request_id,
                    },
                )
                if response.status_code < 500:
                    response.raise_for_status()
            except httpx.TransportError:
                logger.warning("Fault submission outcome is unknown: %s", request.request_id, exc_info=True)
            return await self._read_receipt(client=client, request=request)

    async def _read_receipt(
        self, *, client: httpx.AsyncClient, request: SoakActionRequest, timeout_seconds: float = 30.0
    ) -> dict:
        async with asyncio.timeout(timeout_seconds):
            while True:
                try:
                    response = await client.get(f"{self._base_url}/api/v1/fault-receipts/{request.request_id}")
                    if response.status_code != 404 and response.status_code < 500:
                        response.raise_for_status()
                        if (payload := response.json()) is not None:
                            receipt_type = (
                                FaultDeadlockReceipt
                                if self._failure_mode in {FailureMode.DEADLOCK, FailureMode.THREAD_DEADLOCK}
                                else FaultStopReceipt if self._failure_mode is FailureMode.SIGSTOP else FaultReceipt
                            )
                            receipt = receipt_type.model_validate(payload)
                            if (receipt.request_id, receipt.target, receipt.mode) != (
                                request.request_id,
                                request.fault_target,
                                self._failure_mode,
                            ):
                                raise ValueError("Fault receipt does not match the requested injection")
                            return receipt.model_dump(mode="json")
                except httpx.TransportError:
                    logger.warning("Fault receipt read failed: %s", request.request_id, exc_info=True)
                await asyncio.sleep(0.2)


class DeletePodFaultForm(SoakActionForm):
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

    async def execute(self, request: SoakActionRequest) -> dict:
        pod = _validate_pod_request(
            request=request, form_name=self.name, namespace=self._namespace, release=self._release
        )
        return await delete_observed_pod(pod)


class ExecSigkillFaultForm(SoakActionForm):
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

    @property
    def process_patterns(self) -> dict[str, str]:
        return {self._container: self._process_pattern}

    async def execute(self, request: SoakActionRequest) -> dict:
        return await self._execute_signal(request=request, operation="kill")

    async def _execute_signal(self, *, request: SoakActionRequest, operation: Literal["kill", "stop"]) -> dict:
        pod = _validate_pod_request(
            request=request, form_name=self.name, namespace=self._namespace, release=self._release
        )
        target = pod.process_targets[self._container]
        assert target.pod_uid == pod.uid and target.pattern == self._process_pattern
        result = await run_command(
            [
                "kubectl",
                "exec",
                "--stdin",
                "--namespace",
                pod.namespace,
                pod.name,
                "--container",
                self._container,
                "--",
                "python3",
                "-m",
                "tests.utils.soak.process_target",
                operation,
                request.request_id,
            ],
            timeout_seconds=KUBECTL_TIMEOUT_SECONDS,
            check=False,
            stdin_data=target.model_dump_json(),
        )
        assert result.returncode == 0, (
            f"No process matching {self._process_pattern!r} was confirmed {operation} inside {pod.name} (exit "
            f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}. A crash nobody caused would "
            f"otherwise be counted as one that happened"
        )
        receipt = (
            ProcessExitReceipt.model_validate_json(result.stdout)
            if operation == "kill"
            else ProcessStopReceipt.model_validate_json(result.stdout)
        )
        receipt.validate_for(request_id=request.request_id, target=target)
        return receipt.model_dump(mode="json")


class ExecSigstopFaultForm(ExecSigkillFaultForm):
    @property
    def name(self) -> str:
        return EXEC_SIGSTOP_FORM_NAME

    async def execute(self, request: SoakActionRequest) -> dict:
        return await self._execute_signal(request=request, operation="stop")


def _validate_pod_request(
    *, request: SoakActionRequest, form_name: str, namespace: str, release: str
) -> SoakPodTarget:
    assert request.form_name == form_name, f"Request {request.request_id} names another form: {request.form_name}"
    assert request.pod is not None, f"Request {request.request_id} names no pod"
    assert (
        request.pod.namespace == namespace and request.pod.release == release
    ), f"Request {request.request_id} targets a different release: {request.pod}"
    return request.pod


CellFaultForms = dict[str, list[SoakActionForm]]


def create_cell_fault_forms(*, base_url: str, config: command_utils.ExecuteTrainConfig) -> CellFaultForms:
    actor_kill_forms = _inject_fault_forms(base_url=base_url, failure_modes=FAILURE_MODES)

    match config.cluster_backend:
        case ClusterBackend.RAY:
            return {
                ACTOR_CELL_TYPE: actor_kill_forms,
                ROLLOUT_CELL_TYPE: _inject_fault_forms(
                    base_url=base_url, failure_modes=RAY_ROLLOUT_ENGINE_FAILURE_MODES
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
            exec_sigstop_form = ExecSigstopFaultForm(
                namespace=config.namespace,
                run_id=config.run_id,
                container=ENGINE_CONTAINER_NAME,
                process_pattern=SGLANG_PROCESS_PATTERN,
            )
            return {
                ACTOR_CELL_TYPE: [*actor_kill_forms, delete_pod_form],
                ROLLOUT_CELL_TYPE: [exec_sigkill_form, exec_sigstop_form, delete_pod_form],
            }


def _inject_fault_forms(*, base_url: str, failure_modes: list[FailureMode]) -> list[SoakActionForm]:
    return [InjectFaultForm(base_url=base_url, failure_mode=failure_mode) for failure_mode in failure_modes]


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
