from typing import Literal

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ProcessIdentity(FrozenStrictBaseModel):
    pid: int = Field(gt=1)
    start_ticks: int


class ProcessTarget(FrozenStrictBaseModel):
    pod_uid: str
    boot_id: str
    pid_namespace: str
    init_start_ticks: int
    pattern: str
    processes: list[ProcessIdentity] = Field(min_length=1)


class ProcessSignalReceipt(FrozenStrictBaseModel):
    kind: Literal["process_signal"] = "process_signal"
    request_id: str = Field(min_length=1)
    target: ProcessTarget
    operation: Literal["kill", "stop"]
    signalled_pids: list[int] = Field(min_length=1)

    def validate_for(self, *, request_id: str, target: ProcessTarget, operation: Literal["kill", "stop"]) -> None:
        raise NotImplementedError


def observe_processes(*, pod_uid: str, pattern: str) -> ProcessTarget:
    raise NotImplementedError
