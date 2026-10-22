import contextvars
import dataclasses
import functools
import inspect
import logging
import os
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import TypeAdapter

from miles.utils.audit_utils.event_logger.models import Event, EventBase
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.tracking_utils.structured_log import log_structured, prune_for_log

logger = logging.getLogger(__name__)

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

EVENTS_DIRNAME: str = "events"


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

    @property
    def log_dir(self) -> Path:
        return self._log_dir

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

    def log(
        self,
        event_cls: type[EventBase],
        partial: dict[str, Any],
        *,
        print_log: bool = True,
        include_context: bool = True,
    ) -> None:
        event = event_cls(
            **{
                **partial,
                "timestamp": datetime.now(timezone.utc),
                "source": self._source,
                **(self._context_var.get({}) if include_context else {}),
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
            log_structured(logger.info, tag="audit", op="event", event=type(event).__name__, **payload)

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
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _maybe_with_context(ctx_fn, args=args, kwargs=kwargs):
                    return await method(*args, **kwargs)

            return async_wrapper

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _maybe_with_context(ctx_fn, args=args, kwargs=kwargs):
                return method(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def _maybe_with_context(
    ctx_fn: Callable[..., dict[str, Any]], *, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Generator[None, None, None]:
    if not is_event_logger_initialized():
        yield
        return

    with get_event_logger().with_context(ctx_fn(*args, **kwargs)):
        yield


def read_events(log_dir: Path, *, strict: bool = False) -> list[Event]:
    """Read all JSONL event files from a directory and return parsed events."""
    return EventReader(log_dir, strict=strict).read()


class EventReader:
    """Read the events of one directory repeatedly, parsing only the lines appended since the previous read."""

    def __init__(self, log_dir: Path, *, strict: bool = False) -> None:
        self._log_dir = log_dir
        self._strict = strict
        self._parsed_files: dict[Path, _ParsedFile] = {}

    def read(self) -> list[Event]:
        jsonl_files = sorted(self._log_dir.glob("**/*.jsonl"))
        if not jsonl_files:
            logger.warning("No JSONL files found in %s", self._log_dir)
            return []

        events: list[Event] = []
        parsed_files: dict[Path, _ParsedFile] = {}
        for jsonl_path in jsonl_files:
            parsed, unterminated_tail = self._read_file(jsonl_path)
            parsed_files[jsonl_path] = parsed
            events += parsed.events
            events += unterminated_tail
        self._parsed_files = parsed_files
        return events

    def _read_file(self, path: Path) -> tuple["_ParsedFile", list[Event]]:
        with open(path, "rb") as f:
            stat = os.fstat(f.fileno())
            parsed = self._parsed_files.get(path)
            if parsed is None or not parsed.is_prefix_of(f, stat=stat):
                parsed = _ParsedFile(file_id=(stat.st_dev, stat.st_ino))
            f.seek(parsed.offset)

            for raw_line in f:
                events = self._parse_line(path, raw_line=raw_line, line_num=parsed.num_lines + 1)
                if not raw_line.endswith(b"\n"):
                    return parsed, events
                parsed.append_line(raw_line, events=events)
        return parsed, []

    def _parse_line(self, path: Path, *, raw_line: bytes, line_num: int) -> list[Event]:
        raw_line = raw_line.strip()
        if not raw_line:
            return []
        try:
            return [_event_adapter.validate_json(raw_line)]
        except Exception:
            if self._strict:
                raise
            logger.warning(
                "Failed to parse event at %s:%d",
                path,
                line_num,
                exc_info=True,
            )
            return []


@dataclasses.dataclass
class _ParsedFile:
    file_id: tuple[int, int]
    offset: int = 0
    num_lines: int = 0
    tail: bytes = b""
    events: list[Event] = dataclasses.field(default_factory=list)

    def is_prefix_of(self, f: BinaryIO, *, stat: os.stat_result) -> bool:
        if (stat.st_dev, stat.st_ino) != self.file_id or stat.st_size < self.offset:
            return False
        f.seek(self.offset - len(self.tail))
        return f.read(len(self.tail)) == self.tail

    def append_line(self, raw_line: bytes, *, events: list[Event]) -> None:
        self.offset += len(raw_line)
        self.num_lines += 1
        self.tail = (self.tail + raw_line[-_PREFIX_CHECK_BYTES:])[-_PREFIX_CHECK_BYTES:]
        self.events += events


_PREFIX_CHECK_BYTES: int = 4096
