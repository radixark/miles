import logging
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)


# Stay above ray's worker port range (10002-19999) and below the ephemeral range
# (32768+). A worker only binds the port reserved for it much later -- a gated engine
# not until the first weight update window -- so a port that merely looks free now
# would otherwise be handed to a ray worker long before its owner claims it.
BASE_PORT = 20000

# The trainer picks its torch-distributed master port by probing for a free one rather
# than reserving it here, so this allocator must never hand out a port inside the band
# the trainer probes.
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
        # Distance from the trainer band is an assumption, not a guarantee: the cursor only
        # ever moves up, and a long-lived run that keeps reconfiguring can walk into it.
        # Fail here rather than let two owners agree on the same port much later.
        assert not _overlaps_train_master_band(port, consecutive), (
            f"port allocator reached the trainer master band: {port}..{port + consecutive - 1} "
            f"overlaps {TRAIN_MASTER_PORT_RANGE}"
        )
        self._next_port_of_ip[node_ip] = port + consecutive
        return port


def _overlaps_train_master_band(port: int, consecutive: int) -> bool:
    low, high = TRAIN_MASTER_PORT_RANGE
    return port + consecutive - 1 >= low and port <= high
