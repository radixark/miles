import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from miles.utils.audit_utils.event_logger.logger import get_event_logger
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import inject_fault

logger = logging.getLogger(__name__)

FaultHookName = Literal["trainer_before_all_gather", "trainer_before_weight_send"]
FaultHookStatus = Literal["armed", "scheduled", "cancelled", "expired", "fired", "failed"]
_active_hook: ContextVar[Callable[[FaultHookName], None] | None] = ContextVar("active_fault_hook", default=None)


def reach_fault_hook(hook: FaultHookName) -> None:
    if (callback := _active_hook.get()) is not None:
        callback(hook)


class FaultHookRequest(FrozenStrictBaseModel):
    request_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    hook: FaultHookName
    mode: Literal["sigkill", "exit", "segfault", "sigstop", "deadlock", "thread_deadlock"]
    lifetime_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    delay_ms: float = Field(default=0.0, ge=0, le=300_000, allow_inf_nan=False)
    receipt_url: str | None = None

    @model_validator(mode="after")
    def _validate_delay(self) -> "FaultHookRequest":
        if self.mode == "thread_deadlock" and self.delay_ms > 0:
            raise ValueError("Training-thread deadlock requires immediate hook execution")
        return self


class FaultHookRecord(FrozenStrictBaseModel):
    request: FaultHookRequest
    status: FaultHookStatus
    armed_at: float
    changed_at: float
    reached_at: float | None = None
    due_at: float | None = None
    rollout_id: int | None = None
    attempt: int | None = None
    weight_version: int | None = None


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
        self._timers: dict[str, threading.Timer] = {}

    @contextmanager
    def weight_update_scope(self, *, weight_version: int) -> Iterator[None]:
        def reach(hook: FaultHookName) -> None:
            self.reach(hook=hook, weight_version=weight_version)

        token = _active_hook.set(reach)
        try:
            yield
        finally:
            _active_hook.reset(token)

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
            if any(record.status in {"armed", "scheduled"} for record in self._records.values()):
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
            if record.status in {"armed", "scheduled"}:
                record = self._transition(record=record, status="cancelled")
            return record

    def read(self, *, request_id: str, instance_id: str) -> FaultHookRecord:
        with self._lock:
            self._check_instance(instance_id)
            self._expire()
            return self._records[request_id]

    def reach(
        self,
        *,
        hook: FaultHookName,
        rollout_id: int | None = None,
        attempt: int | None = None,
        weight_version: int | None = None,
    ) -> None:
        with self._lock:
            self._expire()
            record = next(
                (x for x in self._records.values() if x.status == "armed" and x.request.hook == hook),
                None,
            )
            if record is None:
                return
            now = time.monotonic()
            record = record.model_copy(
                update={
                    "rollout_id": rollout_id,
                    "attempt": attempt,
                    "weight_version": weight_version,
                    "reached_at": now,
                    "due_at": now + record.request.delay_ms / 1000,
                }
            )
            if record.request.delay_ms > 0:
                record = self._transition(record=record, status="scheduled")
                timer = threading.Timer(
                    interval=min(record.due_at, record.armed_at + record.request.lifetime_seconds) - now,
                    function=self._fire_scheduled,
                    kwargs={"request_id": record.request.request_id},
                )
                timer.daemon = True
                self._timers[record.request.request_id] = timer
                try:
                    timer.start()
                except Exception:
                    logger.exception("Could not schedule fault hook: %s", record.request.request_id)
                    self._transition(record=record, status="failed")
                    raise
                return
            record = self._transition(record=record, status="fired")

        self._execute(record)

    def _fire_scheduled(self, *, request_id: str) -> None:
        with self._lock:
            self._expire()
            record = self._records[request_id]
            if record.status != "scheduled":
                return
            record = self._transition(record=record, status="fired")

        self._execute(record)

    def _execute(self, record: FaultHookRecord) -> None:
        try:
            inject_fault(
                mode=record.request.mode,
                request_id=record.request.request_id,
                receipt_url=record.request.receipt_url,
            )
            raise RuntimeError("A fault hook unexpectedly returned")
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
            if record.status in {"armed", "scheduled"} and now >= record.armed_at + record.request.lifetime_seconds:
                self._transition(record=record, status="expired")

    def _transition(self, *, record: FaultHookRecord, status: FaultHookStatus) -> FaultHookRecord:
        updated = record.model_copy(update={"status": status, "changed_at": time.monotonic()})
        self._record(updated)
        if status not in {"armed", "scheduled"} and (timer := self._timers.pop(record.request.request_id, None)):
            timer.cancel()
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
                reached_at=record.reached_at,
                due_at=record.due_at,
                rollout_id=record.rollout_id,
                attempt=record.attempt,
                weight_version=record.weight_version,
            ),
        )
        self._records[record.request.request_id] = record
