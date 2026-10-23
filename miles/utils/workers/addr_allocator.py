import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        async with self._lock:
            # The node scans from this cursor within its own service port range and wraps
            # back to the range's start; a node without a cursor yet starts there.
            port: int = await actor._get_free_port_block.remote(
                start_port=self._next_port_of_ip.get(node_ip),
                count=consecutive,
            )
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
