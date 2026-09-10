import logging
import threading
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import inject_fault

logger = logging.getLogger(__name__)

FaultHookName = Literal["trainer_before_all_gather", "trainer_before_weight_send"]
FaultHookStatus = Literal["armed", "cancelled", "expired", "fired", "failed"]


class FaultHookRequest(FrozenStrictBaseModel):
    request_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    hook: FaultHookName
    mode: Literal["sigkill", "exit", "segfault"]
    lifetime_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    receipt_url: str | None = None


class FaultHookRecord(FrozenStrictBaseModel):
    request: FaultHookRequest
    status: FaultHookStatus
    armed_at: float
    changed_at: float
    rollout_id: int | None = None
    attempt: int | None = None


class FaultHookCommand(FrozenStrictBaseModel):
    operation: Literal["inspect", "arm", "cancel", "read"]
    request: FaultHookRequest | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> "FaultHookCommand":
        if (self.operation == "inspect") != (self.request is None):
            raise ValueError("Only hook inspection omits the fault request")
        return self


class FaultHookRegistry:
    def __init__(self) -> None:
        self.instance_id = uuid4().hex
        self._lock = threading.Lock()
        self._records: dict[str, FaultHookRecord] = {}

    def control(self, command: FaultHookCommand) -> str | FaultHookRecord:
        if command.operation == "inspect":
            return self.instance_id
        assert (request := command.request) is not None
        match command.operation:
            case "arm":
                return self.arm(request)
            case "cancel":
                return self.cancel(request)
            case "read":
                return self.read(request_id=request.request_id, instance_id=request.instance_id)
        raise ValueError(f"Unknown fault hook operation: {command.operation}")

    def arm(self, request: FaultHookRequest) -> FaultHookRecord:
        with self._lock:
            self._check_instance(request.instance_id)
            self._expire()
            if (existing := self._records.get(request.request_id)) is not None:
                if existing.request != request:
                    raise ValueError("Fault hook request ID was reused with different parameters")
                return existing
            if any(record.status == "armed" for record in self._records.values()):
                raise ValueError("A fault hook is already armed in this process")
            now = time.monotonic()
            record = FaultHookRecord(request=request, status="armed", armed_at=now, changed_at=now)
            self._record(record)
            return record

    def cancel(self, request: FaultHookRequest) -> FaultHookRecord:
        with self._lock:
            self._check_instance(request.instance_id)
            self._expire()
            if (record := self._records.get(request.request_id)) is None:
                now = time.monotonic()
                record = FaultHookRecord(request=request, status="cancelled", armed_at=now, changed_at=now)
                self._record(record)
                return record
            if record.request != request:
                raise ValueError("Fault hook cancellation does not match the original request")
            if record.status == "armed":
                record = self._transition(record=record, status="cancelled")
            return record

    def read(self, *, request_id: str, instance_id: str) -> FaultHookRecord:
        with self._lock:
            self._check_instance(instance_id)
            self._expire()
            return self._records[request_id]

    def reach(self, *, hook: FaultHookName, rollout_id: int, attempt: int) -> None:
        with self._lock:
            self._expire()
            record = next(
                (x for x in self._records.values() if x.status == "armed" and x.request.hook == hook),
                None,
            )
            if record is None:
                return
            record = record.model_copy(update={"rollout_id": rollout_id, "attempt": attempt})
            record = self._transition(record=record, status="fired")

        try:
            inject_fault(
                mode=record.request.mode,
                request_id=record.request.request_id,
                receipt_url=record.request.receipt_url,
            )
            raise RuntimeError("A terminating fault hook unexpectedly returned")
        except Exception:
            logger.exception("Fault hook execution failed: %s", record.request.request_id)
            with self._lock:
                self._transition(record=record, status="failed")
            raise

    def _check_instance(self, instance_id: str) -> None:
        if instance_id != self.instance_id:
            raise ValueError("Fault hook targets another process incarnation")

    def _expire(self) -> None:
        now = time.monotonic()
        for record in tuple(self._records.values()):
            if record.status == "armed" and now >= record.armed_at + record.request.lifetime_seconds:
                self._transition(record=record, status="expired")

    def _transition(self, *, record: FaultHookRecord, status: FaultHookStatus) -> FaultHookRecord:
        updated = record.model_copy(update={"status": status, "changed_at": time.monotonic()})
        self._record(updated)
        return updated

    def _record(self, record: FaultHookRecord) -> None:
        get_event_logger().log(
            FaultHookEvent,
            dict(
                request_id=record.request.request_id,
                instance_id=record.request.instance_id,
                hook=record.request.hook,
                mode=record.request.mode,
                status=record.status,
                monotonic_time=record.changed_at,
                rollout_id=record.rollout_id,
                attempt=record.attempt,
            ),
        )
        self._records[record.request.request_id] = record
