import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import (
    FaultHookAction,
    FaultHookContext,
    FaultHookEvent,
    FaultHookName,
    FaultHookRecord,
    FaultHookRequest,
    FaultHookStatus,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.fault_injector import inject_fault

logger = logging.getLogger(__name__)


def reach_fault_hook(hook_name: FaultHookName) -> None:
    try:
        fault_hook_controller._reach(hook_name)
    except Exception:
        logger.exception("Could not observe or dispatch fault hook: %s", hook_name)


class FaultHookConflictError(Exception):
    pass


class FaultHookOperation(StrEnum):
    SET = "set"
    CLEAR = "clear"


class FaultHookCommand(FrozenStrictBaseModel):
    operation: FaultHookOperation
    request: FaultHookRequest


class _FaultHookController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ongoing: _FaultHookRequestExecutor | None = None
        self._context: FaultHookContext | None = None

    @contextmanager
    def with_context(self, context: FaultHookContext) -> Iterator[None]:
        self._context = context
        try:
            yield
        finally:
            self._context = None

    def apply(self, command: FaultHookCommand) -> FaultHookRecord:
        with self._lock:
            self._drop_expired()
            match command.operation:
                case FaultHookOperation.SET:
                    return self._set(command.request)
                case FaultHookOperation.CLEAR:
                    return self._clear(command.request)

    def _set(self, request: FaultHookRequest) -> FaultHookRecord:
        if self._ongoing is not None:
            raise FaultHookConflictError("A fault hook is already set in this process")
        self._ongoing = executor = _FaultHookRequestExecutor(request)
        return executor.record

    def _clear(self, request: FaultHookRequest) -> FaultHookRecord:
        if (executor := self._ongoing) is None or executor.record.request.request_id != request.request_id:
            raise FaultHookConflictError("Fault hook clearing names a request this process never set")
        if executor.record.request != request:
            raise FaultHookConflictError("Fault hook clearing does not match the original request")
        self._ongoing = None
        return executor.clear()

    def _reach(self, hook_name: FaultHookName) -> None:
        with self._lock:
            self._drop_expired()
            if (executor := self._ongoing) is None or executor.record.request.hook_name != hook_name:
                return
            self._ongoing = None
            executor.fire(context=self._context)

        executor.execute()

    def _drop_expired(self) -> None:
        if (executor := self._ongoing) is not None and executor.is_expired():
            self._ongoing = None
            executor.expire()


class _FaultHookRequestExecutor:
    def __init__(self, request: FaultHookRequest) -> None:
        now = time.monotonic()
        self.record = FaultHookRecord(request=request, status=FaultHookStatus.PENDING, set_at=now, changed_at=now)
        self._log_event()

    def is_expired(self) -> bool:
        return time.monotonic() >= self.record.set_at + self.record.request.lifetime_seconds

    def clear(self) -> FaultHookRecord:
        return self._transition(FaultHookStatus.CLEARED)

    def expire(self) -> FaultHookRecord:
        return self._transition(FaultHookStatus.EXPIRED)

    def fire(self, *, context: FaultHookContext | None) -> FaultHookRecord:
        self.record = self.record.model_copy(update={"context": context, "reached_at": time.monotonic()})
        return self._transition(FaultHookStatus.FIRED)

    def execute(self) -> None:
        if self.record.request.action == FaultHookAction.OBSERVE:
            return
        try:
            inject_fault(mode=self.record.request.mode.value)
            raise RuntimeError("A fault hook unexpectedly returned")
        except Exception:
            logger.exception("Fault hook execution failed: %s", self.record.request.request_id)
            self._transition(FaultHookStatus.FAILED)

    def _transition(self, status: FaultHookStatus) -> FaultHookRecord:
        self.record = self.record.model_copy(update={"status": status, "changed_at": time.monotonic()})
        self._log_event()
        return self.record

    def _log_event(self) -> None:
        if not is_event_logger_initialized():
            return
        try:
            get_event_logger().log(FaultHookEvent, dict(record=self.record))
        except Exception:
            logger.exception("Could not record fault hook event: %s", self.record.request.request_id)


fault_hook_controller = _FaultHookController()
