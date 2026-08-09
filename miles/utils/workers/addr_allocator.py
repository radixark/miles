import logging
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)


BASE_PORT = 20000

TRAIN_MASTER_PORT_RANGE = (30000, 31000)


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)

    def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        start_port = self._next_port_of_ip.get(node_ip, BASE_PORT)
        port: int = ray.get(
            actor._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
            )
        )
        assert not _overlaps_train_master_band(port, consecutive), (
            f"port allocator reached the trainer master band: {port}..{port + consecutive - 1} "
            f"overlaps {TRAIN_MASTER_PORT_RANGE}"
        )
        self._next_port_of_ip[node_ip] = port + consecutive
        return port


def _overlaps_train_master_band(port: int, consecutive: int) -> bool:
    low, high = TRAIN_MASTER_PORT_RANGE
    return port + consecutive - 1 >= low and port <= high
