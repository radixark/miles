from dataclasses import dataclass

from pydantic import Field, field_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import FailureMode
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


@dataclass
class _Request:
    target: FaultTarget
    mode: FailureMode
    operation_key: str
    receipt: FaultReceipt | None = None


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

    def publish(self, *, request_id: str, submission: FaultExitSubmission) -> FaultReceipt:
        request = self._requests[request_id]
        receipt = FaultReceipt(
            request_id=request_id, target=request.target, mode=request.mode, exited_pids=submission.exited_pids
        )
        if request.receipt is not None and request.receipt != receipt:
            raise ValueError("Conflicting fault receipt")
        request.receipt = receipt
        return receipt

    def read(self, request_id: str) -> FaultReceipt | None:
        return self._requests[request_id].receipt
