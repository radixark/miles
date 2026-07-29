import logging
import multiprocessing
import random
import sys
import uuid

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.server import run_session_server
from miles.router.config import compute_miles_router_config
from miles.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    is_port_available,
    wait_for_server_ready,
)
from miles.utils.workers import process_utils
from miles.utils.workers.argv_utils import config_to_argv

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


def start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sgl router or miles router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set and ``force_new`` is False,
    skip launching and return the existing values.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_miles_router:
        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."

        router_config = compute_miles_router_config(args, host=router_ip, port=router_port)
        launch_argv = [sys.executable, "-m", "miles.router.router", *config_to_argv(router_config)]

    else:
        router_args = compute_sglang_router_args(
            args,
            host=router_ip,
            port=router_port,
            prometheus_port=find_available_port(random.randint(4000, 5000)),
            has_pd_disaggregation=has_pd_disaggregation,
        )
        logger.info(f"Launch router with args: {router_args}")
        launch_argv = [sys.executable, "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)]

    port = router_port
    if not is_port_available(port):
        raise RuntimeError(
            f"Port {port} is already in use — a stale router process may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    process = process_utils.launch_bound_subprocess(launch_argv, envs={})
    wait_for_server_ready(router_ip, router_port, process, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


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
    # spawn (not fork): the child must not inherit threads/finalizers from this
    # Ray actor (e.g. wandb's service thread), which deadlock a forked child.
    instance_ids: dict[int, str] = {}
    processes = []
    for port in ports:
        instance_id = uuid.uuid4().hex
        instance_ids[port] = instance_id
        config = compute_session_server_config(
            args, host=ip, port=port, instance_id=instance_id, backend_url=router_url
        )
        process = multiprocessing.get_context("spawn").Process(target=run_session_server, args=(config,))
        process.daemon = True
        process.start()
        processes.append((port, process))
    # The per-port map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for port, process in processes:
        wait_for_server_ready(ip, port, process, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Session servers launched at {ip}, ports {ports} ({len(ports)} instances)")
