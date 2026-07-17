import logging
import os
import re
import sys
import warnings
from miles.utils.audit_utils.event_logger.logger import EventLogger, is_event_logger_initialized, set_event_logger
from miles.utils.audit_utils.process_identity import ProcessIdentity

_LOGGER_CONFIGURED = False

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"


def configure_logger(args, *, source: ProcessIdentity) -> None:
    name = source.to_name()
    configure_logger_raw(name)

    if (event_dir := getattr(args, "save_debug_event_data", None)) is not None:
        if not is_event_logger_initialized():
            set_event_logger(EventLogger(log_dir=event_dir, file_name=f"{name}.jsonl", source=source))


# ref: SGLang
def configure_logger_raw(name: str = "") -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s.%(msecs)03d {name}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

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
