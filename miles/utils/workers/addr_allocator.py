import asyncio
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _start_port: int = field(
        default_factory=lambda: int(os.environ.get("MILES_DYNAMIC_PORT_START", _DYNAMIC_PORT_START))
    )

    def __post_init__(self):
        if not 1 <= self._start_port <= _MAX_PORT:
            raise ValueError(f"MILES_DYNAMIC_PORT_START must be between 1 and {_MAX_PORT}, got {self._start_port}")

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        async with self._lock:
            # Keep service ports outside Ray's worker range and the host's ephemeral
            # client range. Hosts with a lower ephemeral range can override the base.
            start_port = self._next_port_of_ip.get(node_ip, self._start_port)
            if start_port + consecutive - 1 > _MAX_PORT:
                start_port = self._start_port
            port: int = await actor._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
            )
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
