import logging
import shlex
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_MASTER_LOG_PATH,
    MOONCAKE_MASTER_METRICS_PORT,
    MOONCAKE_MASTER_PORT,
    _is_tcp_server_ready,
    run_shell_command,
    substitute_placeholders,
)
from miles.utils.http_utils import wait_for_server_ready
from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.001)
def _exec_command_on_node(cmd: str, capture_output: bool) -> str | None:
    return run_shell_command(f"unset CUDA_VISIBLE_DEVICES; {cmd}", capture_output=capture_output)


def exec_command_all_ray_nodes(
    cmd: str, capture_output: bool = False, num_nodes: int | None = None
) -> list[str | None]:
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

        refs = []
        for rank, node in enumerate(nodes):
            node_cmd = substitute_placeholders(
                cmd,
                node_rank=str(rank),
                nnodes=nnodes,
                master_addr=master_addr,
                node_ip=node["NodeManagerAddress"],
            )
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


def start_mooncake_master(
    rpc_port: int = MOONCAKE_MASTER_PORT,
    metrics_port: int = MOONCAKE_MASTER_METRICS_PORT,
    timeout: float = 30,
    log_path: str | Path = MOONCAKE_MASTER_LOG_PATH,
) -> None:
    host = "127.0.0.1"
    if _is_tcp_server_ready(host, rpc_port):
        logger.info(f"Mooncake master is already ready at {host}:{rpc_port}")
        return

    log_path = Path(log_path)
    quoted_log_path = shlex.quote(str(log_path))
    run_shell_command(
        "pkill -x mooncake_master >/dev/null 2>&1 || true; "
        f"(setsid mooncake_master --rpc_port {rpc_port} --metrics_port {metrics_port} "
        f"> {quoted_log_path} 2>&1 &)"
    )
    try:
        wait_for_server_ready(host, rpc_port, timeout=timeout)
    except RuntimeError as exc:
        run_shell_command("pkill -x mooncake_master >/dev/null 2>&1 || true")
        try:
            log_lines = log_path.read_text(errors="replace").splitlines()
            log_tail = "\n".join(log_lines[-100:]) or "<empty>"
        except OSError as log_error:
            log_tail = f"<unable to read {log_path}: {log_error}>"
        raise RuntimeError(
            f"Mooncake master at {host}:{rpc_port} did not become ready.\n"
            f"Last 100 lines of {log_path}:\n{log_tail}"
        ) from exc
