import contextvars
import functools
import inspect
import logging
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from miles.utils.audit_utils.event_logger.models import Event, EventBase
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.tracking_utils.structured_log import log_structured, prune_for_log

logger = logging.getLogger(__name__)

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


class EventLogger:
    def __init__(self, *, log_dir: Path | str, file_name: str = "events.jsonl", source: ProcessIdentity) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._path = self._log_dir / file_name
        self._source = source
        self._context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
            "event_logger_context",
        )

    @property
    def source(self) -> ProcessIdentity:
        return self._source

    @contextmanager
    def with_context(self, ctx: dict[str, Any]) -> Generator[None, None, None]:
        """Temporarily merge extra fields into every event logged within this scope.

        Safe for both threads and asyncio tasks (uses contextvars).
        """
        prev = self._context_var.get({})
        merged = {**prev, **ctx}
        token = self._context_var.set(merged)
        try:
            yield
        finally:
            assert self._context_var.get() == merged
            self._context_var.reset(token)

    def log(self, event_cls: type[EventBase], partial: dict[str, Any], *, print_log: bool = True) -> None:
        event = event_cls(
            **{
                **partial,
                "timestamp": datetime.now(timezone.utc),
                "source": self._source,
                **self._context_var.get({}),
            }
        )
        line = event.model_dump_json() + "\n"
        with self._lock:
            # Opened per write so the file can be replaced (e.g. restored from a
            # checkpoint snapshot) at any point between events.
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        if print_log:
            payload = prune_for_log(event.model_dump(mode="json", exclude={"timestamp", "source"}))
            log_structured(logger.info, op="event", event=type(event).__name__, **payload)

    def close(self) -> None:
        pass


_event_logger: EventLogger | None = None


def set_event_logger(event_logger: EventLogger | None) -> None:
    global _event_logger
    _event_logger = event_logger


def get_event_logger() -> EventLogger:
    if _event_logger is None:
        raise RuntimeError("EventLogger not initialized. Call set_event_logger() first.")
    return _event_logger


def is_event_logger_initialized() -> bool:
    return _event_logger is not None


def event_logger_context(ctx_fn: Callable[..., dict[str, Any]]) -> Callable:
    """Decorator that wraps a method with EventLogger.with_context if initialized.

    ``ctx_fn`` receives the same arguments as the decorated method and returns
    the context dict.  If the event logger is not initialized, the method runs
    without any context.
    """

    def decorator(method: Callable) -> Callable:
        assert not inspect.iscoroutinefunction(method), "event_logger_context does not support async methods"

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_event_logger_initialized():
                return method(*args, **kwargs)

            ctx_value = ctx_fn(*args, **kwargs)
            with get_event_logger().with_context(ctx_value):
                return method(*args, **kwargs)

        return wrapper

    return decorator


def read_events(log_dir: Path) -> list[Event]:
    """Read all JSONL event files from a directory and return parsed events."""
    events: list[Event] = []

    jsonl_files = sorted(log_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", log_dir)
        return events

    for jsonl_path in jsonl_files:
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    event = _event_adapter.validate_json(raw_line)
                    events.append(event)
                except Exception:
                    logger.warning(
                        "Failed to parse event at %s:%d",
                        jsonl_path,
                        line_num,
                        exc_info=True,
                    )

    return events
