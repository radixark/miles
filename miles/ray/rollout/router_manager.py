import functools
import logging
import random
import shlex
import sys
import uuid

import ray

from miles.backends.sglang_utils.sglang_config import ModelConfig
from miles.ray.specs.inference import _compute_spec_router
from miles.rollout.session.config import compute_session_server_config
from miles.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, is_port_available, wait_tcp_ready
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.cell_launch import create_head_worker_actor
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.worker_spec import LaunchCommandContext

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


def start_router(args, model_idx: int, model_cfg: ModelConfig) -> tuple[str, int]:
    """Start sgl router or miles router and return (router_ip, router_port)."""
    router_ip = _wrap_ipv6(get_host_info()[1])
    router_port = find_available_port(random.randint(3000, 4000))
    prometheus_port = find_available_port(random.randint(4000, 5000))

    spec = _compute_spec_router(args, model_idx=model_idx, model_cfg=model_cfg)
    ports = dict(primary=router_port, prometheus=prometheus_port)
    assert set(ports) == {info.name for info in spec.port_infos}
    launch_command = spec.launch_command(LaunchCommandContext(host=router_ip, ports=ports))

    actor_handle = _launch_command_on_head(launch_command)
    wait_tcp_ready(
        router_ip,
        router_port,
        is_alive=functools.partial(_actor_is_alive, actor_handle),
        timeout=_SERVER_READY_TIMEOUT_SECS,
    )
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def _launch_command_on_head(launch_cmd: str) -> ray.actor.ActorHandle:
    actor_handle = create_head_worker_actor(worker_cls=CommandActor, env_vars={}, num_cpus=0.2, ctor_kwargs={})
    actor_handle.run.remote(cmd=launch_cmd, envs={})
    return actor_handle


def _actor_is_alive(actor_handle: ray.actor.ActorHandle) -> bool:
    try:
        ray.get(actor_handle._get_node_ip.remote(), timeout=30)
        return True
    except Exception:
        return False


def _resolve_session_server_ports(start: int | None, workers: int) -> list[int]:
    """Return the requested number of consecutive ports from the configured or auto-selected start."""
    if workers < 1:
        raise ValueError("--session-server-workers must be at least 1.")
    # TODO(#1837): Refactor IP/port allocation; keep this naive for now.
    if start is None:
        start = find_available_port(random.randint(5000, 6000))
    return list(range(start, start + workers))


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
    ports = _resolve_session_server_ports(args.session_server_port, args.session_server_workers)
    for port in ports:
        if not is_port_available(port):
            raise RuntimeError(
                f"Port {port} is already in use — a stale session server may still be running. "
                f"Run 'pkill -9 python' to kill it, then retry."
            )
    # The canonical driver-side value; rollout code picks from this list.
    args.session_server_ports = ports

    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

    # Spawn all children before waiting on any: each child pays the ~10s
    # transformers import, so N servers start in ~one import of wall-time.
    instance_ids: dict[int, str] = {}
    launches = []
    for port in ports:
        instance_id = uuid.uuid4().hex
        instance_ids[port] = instance_id
        config = compute_session_server_config(
            args, host=ip, port=port, instance_id=instance_id, backend_url=router_url
        )
        launch_argv = [sys.executable, "-m", "miles.rollout.session.server", *config_to_argv(config)]
        launches.append((port, _launch_command_on_head(shlex.join(launch_argv))))
    # The per-port map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for port, actor_handle in launches:
        wait_tcp_ready(
            ip, port, is_alive=functools.partial(_actor_is_alive, actor_handle), timeout=_SERVER_READY_TIMEOUT_SECS
        )
    logger.info(f"Session servers launched at {ip}, ports {ports} ({len(ports)} instances)")
