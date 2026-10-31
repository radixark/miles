"""Port ranges that stay outside the host's ephemeral port range.

Miles probes a free port, passes its number to the process that serves on it, and that
process binds it seconds or minutes later. A port inside the kernel's ephemeral range
(``net.ipv4.ip_local_port_range``) can be taken in between as the source port of an
outgoing connection, and the bind then fails with "Address already in use".
"""

import os
from pathlib import Path

_EPHEMERAL_PORT_RANGE_FILE = Path("/proc/sys/net/ipv4/ip_local_port_range")
_DYNAMIC_PORT_START_ENV = "MILES_DYNAMIC_PORT_START"
_DEFAULT_SERVICE_PORT_START = 20000
_MIN_UNPRIVILEGED_PORT = 1024
_MAX_PORT = 65535
# Ray hands its workers ports from this range unless `ray start` is given another one.
_RAY_DEFAULT_WORKER_PORTS = (10002, 19999)


def ephemeral_port_range() -> tuple[int, int] | None:
    """The kernel's inclusive ephemeral port range on this host, or None where it cannot be read."""
    try:
        low, high = (int(field) for field in _EPHEMERAL_PORT_RANGE_FILE.read_text().split())
    except (OSError, ValueError):
        return None
    return low, high


def service_port_range() -> tuple[int, int]:
    """The inclusive range Miles allocates its service ports from on this host.

    It starts at ``MILES_DYNAMIC_PORT_START`` when that is set, else at 20000, above Ray's
    default worker ports, and ends just below the ephemeral range when that lies above the
    start. When 20000 itself lies inside the ephemeral range, the range moves to the wider
    of the stretches outside it: below both the ephemeral range and Ray's default worker
    ports, or above them.
    """
    ephemeral = ephemeral_port_range()
    configured = os.environ.get(_DYNAMIC_PORT_START_ENV)
    if configured is not None:
        return _range_from(start=_validated_port(configured), ephemeral=ephemeral)
    if ephemeral is None or not ephemeral[0] <= _DEFAULT_SERVICE_PORT_START <= ephemeral[1]:
        return _range_from(start=_DEFAULT_SERVICE_PORT_START, ephemeral=ephemeral)
    return _widest_range_outside(ephemeral)


def ray_worker_port_args() -> str:
    """Flags for ``ray start`` that keep Ray's worker ports out of the ephemeral range.

    Ray hands each worker a port from its worker range and the worker binds it as it starts,
    which leaves the same gap. Where the default range overlaps the ephemeral range, port 0
    makes every worker bind a port the kernel assigns at bind time; elsewhere the default
    range is kept. The result is empty or starts with a space, so it can be appended to a
    command line.
    """
    ephemeral = ephemeral_port_range()
    if ephemeral is None or not _overlaps(ephemeral, _RAY_DEFAULT_WORKER_PORTS):
        return ""
    return " --min-worker-port 0 --max-worker-port 0"


def _validated_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= _MAX_PORT:
        raise ValueError(f"{_DYNAMIC_PORT_START_ENV} must be between 1 and {_MAX_PORT}, got {port}")
    return port


def _range_from(*, start: int, ephemeral: tuple[int, int] | None) -> tuple[int, int]:
    if ephemeral is not None and start < ephemeral[0]:
        return start, ephemeral[0] - 1
    return start, _MAX_PORT


def _widest_range_outside(ephemeral: tuple[int, int]) -> tuple[int, int]:
    below = (_MIN_UNPRIVILEGED_PORT, min(ephemeral[0], _RAY_DEFAULT_WORKER_PORTS[0]) - 1)
    above = (max(ephemeral[1], _RAY_DEFAULT_WORKER_PORTS[1]) + 1, _MAX_PORT)
    candidates = [(low, high) for low, high in (below, above) if low <= high]
    if not candidates:
        return _DEFAULT_SERVICE_PORT_START, _MAX_PORT
    return max(candidates, key=lambda candidate: candidate[1] - candidate[0])


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]
