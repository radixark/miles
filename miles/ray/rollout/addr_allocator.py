import functools
import logging
from dataclasses import dataclass

import ray

logger = logging.getLogger(__name__)


@dataclass
class PortAllocator:
    _values: dict[str, int]

    @staticmethod
    def empty() -> "PortAllocator":
        return PortAllocator(_values={})

    def alloc(self, *, engine, node_ip: str, consecutive: int = 1) -> int:
        # use small ports to prevent ephemeral port between 32768 and 65536.
        # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
        start_port = self._values.get(node_ip, 15000)
        _, port = ray.get(
            engine._get_current_node_ip_and_free_port.remote(
                start_port=start_port,
                consecutive=consecutive,
            )
        )
        self._values[node_ip] = port + consecutive
        return port


# NOTE: May re-implement this in a potentially easier way if needed
def allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    port_allocator: PortAllocator,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        node_ip, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())

        get_port = functools.partial(port_allocator.alloc, engine=engine, node_ip=node_ip)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = node_ip
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()
            # Always allocate a unique engine_info_bootstrap_port per engine
            addr_and_ports[current_rank]["engine_info_bootstrap_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{node_ip}:{get_port(consecutive=30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i][
                    "dist_init_addr"
                ] = f"{node_ip}:{get_port(consecutive=30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports
