import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    # cursors of dedicated ranges (see PortInfo.dynamic_start), keyed by (node_ip, range_start)
    _next_port_of_ip_by_range: dict[tuple[str, int], int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1, start_port: int | None = None) -> int:
        """Allocate ``consecutive`` free ports on ``node_ip``.

        ``start_port`` selects a dedicated range with its own per-node cursor instead of the shared
        per-node range that starts at ``_DYNAMIC_PORT_START`` -- for ports that must not collide with
        anything the shared range hands out on *other* nodes (a multi-node engine's dist_init block).
        """
        async with self._lock:
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            if start_port is None:
                range_start = _DYNAMIC_PORT_START
                cursor = self._next_port_of_ip.get(node_ip, range_start)
            else:
                range_start = start_port
                cursor = self._next_port_of_ip_by_range.get((node_ip, range_start), range_start)
            if cursor + consecutive - 1 > _MAX_PORT:
                cursor = range_start
            port: int = await actor._get_free_port_block.remote(
                start_port=cursor,
                count=consecutive,
            )
            if start_port is None:
                self._next_port_of_ip[node_ip] = port + consecutive
            else:
                self._next_port_of_ip_by_range[(node_ip, range_start)] = port + consecutive
            return port
