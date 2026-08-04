import logging
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)


# Stay above ray's worker port range (10002-19999) and below the ephemeral range
# (32768+). A worker only binds the port reserved for it much later -- a gated engine
# not until the first weight update window -- so a port that merely looks free now
# would otherwise be handed to a ray worker long before its owner claims it.
BASE_PORT = 20000


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
        self._next_port_of_ip[node_ip] = port + consecutive
        return port
