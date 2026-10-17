import asyncio
import random
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.types import SoakActionEvidence, SoakActionRequest
from tests.utils.soak.ft.actions.base import BaseCellFaultForm
from tests.utils.soak.ft.types import CellTarget, PodDetails
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget, delete_observed_pod
from tests.utils.soak.k8s_utils.pod_processes import ProcessSignal, ProcessSignalReceipt

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.test_utils.kubectl_reads import KUBECTL_TIMEOUT_SECONDS
from miles.utils.workers.types import DeployComponent

DELETE_POD_FORM_NAME: str = "delete_pod"
EXEC_SIGKILL_FORM_NAME: str = "exec_sigkill"
EXEC_SIGSTOP_FORM_NAME: str = "exec_sigstop"

ENGINE_CONTAINER_NAME: str = "engine"
SGLANG_PROCESS_PATTERN: str = "sglang::"


@dataclass(frozen=True, kw_only=True)
class BasePodFaultForm(BaseCellFaultForm):
    namespace: str
    run_id: str

    def __post_init__(self) -> None:
        assert self.namespace, "Harming a cell's pod needs the namespace the run was installed into"
        assert self.run_id, "Harming a cell's pod needs the run_id naming the release that owns it"

    @property
    def release(self) -> str:
        return ReleaseName(
            run_id=self.run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None
        ).serialize()

    def maybe_create_request(
        self,
        *,
        target: CellTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None:
        candidates = [
            pod
            for pod in target.pods
            if all(
                container in pod.process_targets and pod.process_targets[container].pattern == pattern
                for container, pattern in self.process_patterns.items()
            )
        ]
        if not candidates:
            return None
        return self._create_request(target=target, details=PodDetails(pod=rng.choice(candidates)))

    def _read_pod(self, request: SoakActionRequest) -> SoakPodTarget:
        assert isinstance(request.details, PodDetails), f"Request {request.request_id} names no pod"
        pod = request.details.pod
        assert (
            pod.namespace == self.namespace and pod.release == self.release
        ), f"Request {request.request_id} targets a different release: {pod}"
        return pod


class DeletePodFaultForm(BasePodFaultForm):
    @property
    def name(self) -> str:
        return DELETE_POD_FORM_NAME

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        report_applied(await delete_observed_pod(self._read_pod(request)))


class BaseExecSignalFaultForm(BasePodFaultForm):
    @property
    @abstractmethod
    def operation(self) -> ProcessSignal: ...

    @property
    def process_patterns(self) -> dict[str, str]:
        return {ENGINE_CONTAINER_NAME: SGLANG_PROCESS_PATTERN}

    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None:
        operation = self.operation
        pod = self._read_pod(request)
        target = pod.process_targets[ENGINE_CONTAINER_NAME]
        assert target.pod_uid == pod.uid and target.pattern == SGLANG_PROCESS_PATTERN
        result = await asyncio.to_thread(
            run_process,
            [
                "kubectl",
                "exec",
                "--stdin",
                "--namespace",
                pod.namespace,
                pod.name,
                "--container",
                ENGINE_CONTAINER_NAME,
                "--",
                "python3",
                "-m",
                "tests.utils.soak.k8s_utils.pod_processes_cli",
                operation,
                request.request_id,
            ],
            capture_output=True,
            check=False,
            input=target.model_dump_json(),
            timeout=KUBECTL_TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, (
            f"No process matching {SGLANG_PROCESS_PATTERN!r} was confirmed {operation} inside {pod.name} (exit "
            f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}. A crash nobody caused would "
            f"otherwise be counted as one that happened"
        )
        receipt = ProcessSignalReceipt.model_validate_json(result.stdout)
        receipt.validate_for(request_id=request.request_id, target=target, operation=operation)
        report_applied(receipt)


class ExecSigkillFaultForm(BaseExecSignalFaultForm):
    @property
    def name(self) -> str:
        return EXEC_SIGKILL_FORM_NAME

    @property
    def operation(self) -> ProcessSignal:
        return ProcessSignal.KILL


class ExecSigstopFaultForm(BaseExecSignalFaultForm):
    @property
    def name(self) -> str:
        return EXEC_SIGSTOP_FORM_NAME

    @property
    def operation(self) -> ProcessSignal:
        return ProcessSignal.STOP
