import logging
import threading
import time
from collections.abc import Callable

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import FaultHookEvent
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, FaultHookRequest, FaultHookStatus

logger = logging.getLogger(__name__)


class FaultHookRequestExecutor:
    def __init__(self, request: FaultHookRequest) -> None:
        now = time.monotonic()
        self._record = FaultHookRecord(request=request, status=FaultHookStatus.PENDING, set_at=now, changed_at=now)
        self._timer: threading.Timer | None = None
        self._log_event()

    @property
    def record(self) -> FaultHookRecord:
        return self._record

    def clear(self) -> FaultHookRecord:
        self._cancel_timer()
        return self._transition(FaultHookStatus.CLEARED)

    def mark_reached(self, *, context: FaultHookContext) -> None:
        now = time.monotonic()
        self._record = self._record.model_copy(
            update={"context": context, "reached_at": now, "due_at": now + self._record.request.delay_ms / 1000}
        )

    def schedule(self, *, on_due: Callable[["FaultHookRequestExecutor"], None]) -> FaultHookRecord:
        assert self._record.due_at is not None
        self._timer = threading.Timer(
            interval=self._record.due_at - time.monotonic(), function=on_due, kwargs={"executor": self}
        )
        self._timer.daemon = True
        self._timer.start()
        return self._transition(FaultHookStatus.SCHEDULED)

    def mark_fired(self) -> "FaultHookRequestExecutor":
        self._transition(FaultHookStatus.FIRED)
        return self

    async def execute(self, *, resources: FaultHookResources) -> None:
        assert self._record.context is not None
        try:
            await self._record.request.action(context=self._record.context, resources=resources)
        except Exception:
            logger.exception("Fault hook execution failed: %s", self._record.request.request_id)
            self._transition(FaultHookStatus.FAILED)
            raise

    def _cancel_timer(self) -> None:
        if (timer := self._timer) is not None:
            self._timer = None
            timer.cancel()

    def _transition(self, status: FaultHookStatus) -> FaultHookRecord:
        self._record = self._record.model_copy(update={"status": status, "changed_at": time.monotonic()})
        self._log_event()
        return self._record

    def _log_event(self) -> None:
        if not is_event_logger_initialized():
            return
        try:
            get_event_logger().log(FaultHookEvent, dict(record=self._record))
        except Exception:
            logger.exception("Could not record fault hook event: %s", self._record.request.request_id)
