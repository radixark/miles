import logging
import os
import re
import sys
import time
import warnings

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED
from miles.utils.audit_utils.event_logger.logger import EventLogger, is_event_logger_initialized, set_event_logger
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.env_report.reporter import EnvReporter, start_env_reporting

_STRICT_ASYNC_WARNINGS_INSTALLED = False
_ENV_REPORTER: EnvReporter | None = None

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"


def configure_logger(args, *, source: ProcessIdentity, report_env: bool = True) -> None:
    name = source.to_name()
    configure_logger_raw(name)
    if SNAPSHOT_TESTING_ENABLED:
        ConfigSnapshotDumper.configure(args=args, source=source)
        ConfigSnapshotDumper.dump(stage="process_config", config={"args": args})

    if (event_dir := args.save_debug_event_data) is not None:
        if not is_event_logger_initialized():
            set_event_logger(EventLogger(log_dir=event_dir, file_name=f"{name}.jsonl", source=source))

    global _ENV_REPORTER
    if report_env and _ENV_REPORTER is None:
        _ENV_REPORTER = start_env_reporting(args)


def rebind_env_reporting(args) -> None:
    # TODO: stop rebinding here; the args object should not be overwritten in place all over the codebase.
    if _ENV_REPORTER is not None:
        _ENV_REPORTER.rebind(args)


# ref: SGLang
def configure_logger_raw(name: str = "") -> None:
    global _STRICT_ASYNC_WARNINGS_INSTALLED

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s.%(msecs)03d {name}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if not _STRICT_ASYNC_WARNINGS_INSTALLED:
        _STRICT_ASYNC_WARNINGS_INSTALLED = True
        configure_strict_async_warnings()


def configure_strict_async_warnings() -> None:
    """Turn unawaited-coroutine warnings into fatal errors.

    Python emits RuntimeWarning when a coroutine is called but never awaited.
    The warning fires inside __del__, so the resulting exception is swallowed
    by sys.unraisablehook. We override the hook to hard-exit the process.
    """
    warnings.filterwarnings("error", category=RuntimeWarning, message=_FATAL_ASYNC_PATTERN)

    _original_hook = sys.unraisablehook

    def _crash_on_async_misuse(unraisable):
        if isinstance(unraisable.exc_value, RuntimeWarning) and re.search(
            _FATAL_ASYNC_PATTERN, str(unraisable.exc_value)
        ):
            msg = f"Fatal async misuse, aborting: {unraisable.exc_value}"
            logger.error(msg)
            print(msg, file=sys.stderr, flush=True)
            os._exit(1)
        _original_hook(unraisable)

    sys.unraisablehook = _crash_on_async_misuse


class RateLimitedWarner:
    """Warn at most once per ``interval_seconds``.

    Background loops (scraper ticks, NVML sampling) hit the same failure
    thousands of times during an outage; observability must neither spam the
    training logs nor die silently, so the first warning goes through and
    repeats are suppressed for the window.
    """

    def __init__(self, logger: logging.Logger, *, interval_seconds: float = 300.0):
        self._logger = logger
        self.interval_seconds = interval_seconds
        self._last_warn = float("-inf")

    def warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warn >= self.interval_seconds:
            self._last_warn = now
            self._logger.warning("%s (further warnings suppressed for %.0fs)", message, self.interval_seconds)

    def reset_window_for_test(self) -> None:
        self._last_warn = float("-inf")
