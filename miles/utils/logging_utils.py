import logging
import os
import re
import sys
import warnings

_LOGGER_CONFIGURED = False

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"


def resolve_log_level() -> int:
    level_name = os.environ.get("MILES_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        logger.warning("Unknown MILES_LOG_LEVEL=%r, falling back to INFO", level_name)
        return logging.INFO
    return level


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    log_level = resolve_log_level()
    logging.basicConfig(
        level=log_level,
        format=f"[%(asctime)s{prefix}] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
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
