import datetime
import logging
import os
import re
import sys
import threading
import warnings
from collections.abc import Callable, Iterable
from typing import TextIO

_LOGGER_CONFIGURED = False

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"

_cached_dist_rank: int | None = None


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
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


def cached_dist_rank() -> int | None:
    """Return the torch.distributed rank, cached once the process group is initialized.

    Before the process group exists, fall back to the RANK env var (Ray train actors set it
    in __init__); return None when neither source is available.
    """
    global _cached_dist_rank
    if _cached_dist_rank is not None:
        return _cached_dist_rank

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        _cached_dist_rank = dist.get_rank()
        return _cached_dist_rank

    env_rank = os.environ.get("RANK")
    return int(env_rank) if env_rank is not None else None


def get_ray_friendly_repr() -> str:
    """Build the per-line tag injected into actor output: torch dist rank + current ms time."""
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    rank = cached_dist_rank()
    rank_repr = "?" if rank is None else str(rank)
    return f"rank={rank_repr} {now}"


class LinePrefixingStream:
    """Text stream wrapper that prepends a freshly-computed tag at the start of every line.

    Ray only prefixes actor logs with ``(ClassName pid=...)``; wrapping stdout/stderr lets us
    inject rank and current time (with ms) into each line, including bare ``print`` output. The
    tag is computed per line so the timestamp is the real emit time, not a value cached at actor
    creation (Ray evaluates an Actor's __repr__ only once and reuses the result forever).
    """

    def __init__(self, underlying: TextIO, build_prefix: Callable[[], str]) -> None:
        self._underlying = underlying
        self._build_prefix = build_prefix
        self._at_line_start = True
        # Reentrant so writelines() can hold the lock while delegating to write().
        self._lock = threading.RLock()

    def write(self, text: str) -> int:
        if not text:
            return 0

        with self._lock:
            segments = text.split("\n")
            for index, segment in enumerate(segments):
                is_last = index == len(segments) - 1
                if segment and self._at_line_start:
                    self._underlying.write(f"[{self._build_prefix()}] ")
                    self._at_line_start = False
                if segment:
                    self._underlying.write(segment)
                if not is_last:
                    self._underlying.write("\n")
                    self._at_line_start = True

        return len(text)

    def writelines(self, lines: Iterable[str]) -> None:
        with self._lock:
            for line in lines:
                self.write(line)

    def flush(self) -> None:
        self._underlying.flush()

    def __getattr__(self, name: str):
        return getattr(self._underlying, name)


def install_ray_actor_log_prefix() -> None:
    """Wrap stdout/stderr so every line gets a ``[rank=N HH:MM:SS.mmm]`` prefix.

    Idempotent. Call after configure_logger so the logging StreamHandler keeps its reference to
    the original stderr and logger lines are not prefixed twice (only bare prints are wrapped).
    """
    if not isinstance(sys.stdout, LinePrefixingStream):
        sys.stdout = LinePrefixingStream(sys.stdout, get_ray_friendly_repr)
    if not isinstance(sys.stderr, LinePrefixingStream):
        sys.stderr = LinePrefixingStream(sys.stderr, get_ray_friendly_repr)
