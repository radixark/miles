import logging
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)


@dataclass
class PortAllocator:
    _values: dict[str, int] = field(default_factory=dict)

    def alloc(self, *, engine, node_ip: str, consecutive: int = 1) -> int:
        # use small ports to prevent ephemeral port between 32768 and 65536.
        # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
        start_port = self._values.get(node_ip, 15000)
        port = ray.get(
            engine._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
            )
        )
        self._values[node_ip] = port + consecutive
        return port
