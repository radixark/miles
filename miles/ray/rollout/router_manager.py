import logging

import ray

from miles.ray.specs.inference import compute_router_pool_id, compute_session_server_instance_id, spec_session_server
from miles.rollout.session.ports import compute_num_session_server_ports, resolve_session_server_ports
from miles.utils.http_utils import is_port_available, wait_tcp_ready
from miles.utils.workers.cell_launch import create_head_worker_actor
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


async def wait_router_ready(model_idx: int) -> HostAndPort:
    """Wait until the model's router, launched by the RayWorkerManager, is reachable and return its address."""
    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    worker_name = compute_worker_name(
        pool_id=compute_router_pool_id(model_idx), cell_index=0, worker_in_cell_index=0
    )
    router_addr = await provider.get_addr(worker_name=worker_name)
    wait_tcp_ready(router_addr.host, router_addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router ready at {router_addr}")
    return router_addr


def _launch_command_on_head(launch_cmd: str) -> ray.actor.ActorHandle:
    actor_handle = create_head_worker_actor(worker_cls=CommandActor, env_vars={}, num_cpus=0.2, ctor_kwargs={})
    actor_handle.run.remote(cmd=launch_cmd, envs={})
    return actor_handle


def start_session_server(args):
    """Start the standalone session servers when ``--use-session-server`` is set.

    One independent single-process server per resolved port; the rollout side
    picks one per session and its URL carries the affinity from then on.
    Always started standalone regardless of whether ``--use-miles-router`` is
    active.
    """
    if not getattr(args, "use_session_server", False):
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    if getattr(args, "session_server_ip", None) is None:
        args.session_server_ip = args.sglang_router_ip

    ip = args.session_server_ip
    ports = resolve_session_server_ports(getattr(args, "session_server_port", None))
    assert len(ports) == compute_num_session_server_ports(args)
    for port in ports:
        if not is_port_available(port):
            raise RuntimeError(
                f"Port {port} is already in use — a stale session server may still be running. "
                f"Run 'pkill -9 python' to kill it, then retry."
            )
    # The canonical driver-side value; rollout code picks from this list.
    args.session_server_ports = ports

    spec = spec_session_server(args)

    # Spawn all children before waiting on any: each child pays the ~10s
    # transformers import, so N servers start in ~one import of wall-time.
    instance_ids: dict[int, str] = {}
    launches = []
    for instance_index, port in enumerate(ports):
        launch_cmd = spec.launch_command(
            LaunchCommandContext(
                cell_index=instance_index,
                worker_in_cell_index=0,
                gpu_ids=[],
                self_addrs=dict(primary=HostAndPort(host=ip, port=port)),
                spec_addrs={
                    compute_router_pool_id(0): [
                        dict(primary=HostAndPort(host=args.sglang_router_ip, port=args.sglang_router_port))
                    ]
                },
            )
        )
        instance_ids[port] = compute_session_server_instance_id(args, instance_index)
        launches.append((port, _launch_command_on_head(launch_cmd)))
    # The per-port map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for port, _ in launches:
        wait_tcp_ready(ip, port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Session servers launched at {ip}, ports {ports} ({len(ports)} instances)")
