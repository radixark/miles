import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # probe_free_port_block(start_port=..., count=...) runs on the node and answers the first free block from there
    async def alloc(
        self, probe_free_port_block: Callable[..., Awaitable[int]], *, node_ip: str, consecutive: int = 1
    ) -> int:
        async with self._lock:
            # The node scans from this cursor within its own service port range and wraps
            # back to the range's start; a node without a cursor yet starts there.
            port: int = await probe_free_port_block(start_port=self._next_port_of_ip.get(node_ip), count=consecutive)
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
