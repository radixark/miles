from dataclasses import dataclass

from pydantic import Field, field_validator, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.test_utils.fault_witness import DeadlockTarget, deadlock_evidence
from miles.utils.workers.cell_operations.base import FaultTarget


class FaultExitSubmission(FrozenStrictBaseModel):
    exited_pids: list[int] = Field(min_length=1)

    @field_validator("exited_pids")
    @classmethod
    def _validate_pids(cls, value: list[int]) -> list[int]:
        if any(pid <= 0 for pid in value) or len(set(value)) != len(value):
            raise ValueError("Exited PIDs must be positive and unique")
        return value


class FaultReceipt(FaultExitSubmission):
    request_id: str = Field(min_length=1)
    target: FaultTarget
    mode: FailureMode


class FaultStopSubmission(FrozenStrictBaseModel):
    stopped_pids: list[int] = Field(min_length=1)

    @field_validator("stopped_pids")
    @classmethod
    def _validate_pids(cls, value: list[int]) -> list[int]:
        if any(pid <= 0 for pid in value) or len(set(value)) != len(value):
            raise ValueError("Stopped PIDs must be positive and unique")
        return value


class FaultDeadlockSubmission(FrozenStrictBaseModel):
    blocked_pid: int = Field(gt=0)
    blocked_tid: int = Field(gt=0)
    lock_device: int = Field(ge=0)
    lock_inode: int = Field(gt=0)
    holds_gil: bool = Field(strict=True)
    lock_evidence: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_lock_evidence(self) -> "FaultDeadlockSubmission":
        target = DeadlockTarget(
            thread_id=self.blocked_tid, device=self.lock_device, inode=self.lock_inode, holds_gil=self.holds_gil
        )
        if deadlock_evidence("\n".join(self.lock_evidence), pid=self.blocked_pid, target=target) != self.lock_evidence:
            raise ValueError("Deadlock receipt does not contain the matching held and waiting locks")
        return self


class FaultDeadlockReceipt(FaultDeadlockSubmission):
    request_id: str = Field(min_length=1)
    target: FaultTarget
    mode: FailureMode


class FaultStopReceipt(FaultStopSubmission):
    request_id: str = Field(min_length=1)
    target: FaultTarget
    mode: FailureMode


@dataclass
class _Request:
    target: FaultTarget
    mode: FailureMode
    operation_key: str
    receipt: FaultReceipt | FaultStopReceipt | FaultDeadlockReceipt | None = None


class FaultReceiptRegistry:
    def __init__(self) -> None:
        self._requests: dict[str, _Request] = {}

    def register(
        self, *, request_id: str, target: FaultTarget, mode: FailureMode, operation_key: str = "immediate"
    ) -> bool:
        if not request_id:
            raise ValueError("Fault request ID must not be empty")
        if (existing := self._requests.get(request_id)) is not None:
            if (existing.target, existing.mode, existing.operation_key) != (target, mode, operation_key):
                raise ValueError("Fault request ID was reused for another injection")
            return False
        self._requests[request_id] = _Request(target=target, mode=mode, operation_key=operation_key)
        return True

    def publish(
        self, *, request_id: str, submission: FaultExitSubmission | FaultStopSubmission | FaultDeadlockSubmission
    ) -> FaultReceipt | FaultStopReceipt | FaultDeadlockReceipt:
        request = self._requests[request_id]
        receipt: FaultReceipt | FaultStopReceipt | FaultDeadlockReceipt
        if isinstance(submission, FaultDeadlockSubmission):
            mode = FailureMode.DEADLOCK if submission.holds_gil else FailureMode.THREAD_DEADLOCK
            if request.mode is not mode:
                raise ValueError("Deadlock receipt has a different blocking scope than the request")
            receipt = FaultDeadlockReceipt(
                request_id=request_id, target=request.target, mode=request.mode, **submission.model_dump()
            )
        elif isinstance(submission, FaultStopSubmission):
            if request.mode is not FailureMode.SIGSTOP:
                raise ValueError("A stop receipt requires a process-stop request")
            receipt = FaultStopReceipt(
                request_id=request_id, target=request.target, mode=request.mode, stopped_pids=submission.stopped_pids
            )
        else:
            if request.mode not in {FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT}:
                raise ValueError("An exit receipt requires a terminating request")
            receipt = FaultReceipt(
                request_id=request_id, target=request.target, mode=request.mode, exited_pids=submission.exited_pids
            )
        if request.receipt is not None and request.receipt != receipt:
            raise ValueError("Conflicting fault receipt")
        request.receipt = receipt
        return receipt

    def read(self, request_id: str) -> FaultReceipt | FaultStopReceipt | FaultDeadlockReceipt | None:
        return self._requests[request_id].receipt
