import logging
import re
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.logging_utils import configure_logger_raw
from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)


def exec_command_gpu(cmd: str, capture_output: bool = False) -> str | None:
    return _exec_command(cmd, capture_output=capture_output)


def exec_command_cpu(cmd: str, capture_output: bool = False) -> str | None:
    return _exec_command(cmd, capture_output=capture_output)


def _exec_command(cmd: str, capture_output: bool = False) -> str | None:
    configure_logger_raw("launcher")
    logger.info(f"EXEC: {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            shell=False,
            check=True,
            capture_output=capture_output,
            **(dict(text=True) if capture_output else {}),
        )
    except subprocess.CalledProcessError as e:
        if capture_output:
            logger.error(f"{e.stdout=} {e.stderr=}")
        raise

    if capture_output:
        logger.info(f"Captured stdout={result.stdout} stderr={result.stderr}")
        return result.stdout
    return None


@ray.remote(num_cpus=0.001)
def _exec_command_on_node(cmd: str, capture_output: bool) -> str | None:
    return _exec_command(f"unset CUDA_VISIBLE_DEVICES; {cmd}", capture_output=capture_output)


def exec_command_multi_node(cmd: str, capture_output: bool = False, num_nodes: int | None = None) -> list[str | None]:
    """Execute a shell command on every alive Ray node in parallel.

    Supported placeholders in `cmd` (replaced per-node before execution):
        {{node_rank}}   - 0-based index of the node
        {{nnodes}}      - total number of alive nodes (or num_nodes if specified)
        {{master_addr}} - NodeManagerAddress of the first node
        {{node_ip}}     - NodeManagerAddress of the current node

    Args:
        num_nodes: If set, only use the first `num_nodes` nodes instead of all alive nodes.
    """
    ray.init(address="auto")
    try:
        current_ip = get_current_node_ip()
        nodes = sorted(
            [n for n in ray.nodes() if n.get("Alive")],
            key=lambda n: (n["NodeManagerAddress"] != current_ip, n["NodeManagerAddress"]),
        )
        assert len(nodes) > 0

        if num_nodes is not None:
            assert num_nodes <= len(nodes), f"Requested {num_nodes} nodes but only {len(nodes)} alive nodes available."
            nodes = nodes[:num_nodes]

        master_addr = nodes[0]["NodeManagerAddress"]
        nnodes = str(len(nodes))

        placeholder_pattern = re.compile(
            "|".join(map(re.escape, ["{{node_rank}}", "{{nnodes}}", "{{master_addr}}", "{{node_ip}}"]))
        )

        refs = []
        for rank, node in enumerate(nodes):
            substitutions = {
                "{{node_rank}}": str(rank),
                "{{nnodes}}": nnodes,
                "{{master_addr}}": master_addr,
                "{{node_ip}}": node["NodeManagerAddress"],
            }
            node_cmd = placeholder_pattern.sub(lambda m, s=substitutions: s[m.group(0)], cmd)
            refs.append(
                _exec_command_on_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    ),
                ).remote(node_cmd, capture_output=capture_output)
            )
        return ray.get(refs)
    finally:
        ray.shutdown()
