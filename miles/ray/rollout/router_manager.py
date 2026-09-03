import copy
import logging
import multiprocessing
import random
import uuid

from sglang_router.launch_router import RouterArgs

from miles.rollout.session.server import run_session_server
from miles.router.router import run_router as run_miles_router
from miles.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, is_port_available
from miles.utils.http_utils import run_router as run_sglang_router
from miles.utils.http_utils import wait_for_server_ready

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

    # Hand off to the Dynamo frontend launcher when that backend is active.
    # It returns the same ``(ip, port)`` shape so the rest of this function
    # never needs to know which router actually came up.
    if getattr(args, "rollout_backend", "sglang") == "dynamo":
        from miles.backends.dynamo_utils.dynamo_router import start_dynamo_router

        return start_dynamo_router(
            args,
            has_pd_disaggregation=has_pd_disaggregation,
            force_new=force_new,
        )

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_miles_router:
        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."

        run_router = run_miles_router
        router_args = copy.copy(args)
        router_args.sglang_router_ip = router_ip
        router_args.sglang_router_port = router_port

    else:
        run_router = run_sglang_router
        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = router_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if args.sglang_router_policy:
            router_args.policy = args.sglang_router_policy

        if has_pd_disaggregation:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    port = router_port
    if not is_port_available(port):
        raise RuntimeError(
            f"Port {port} is already in use — a stale router process may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    # spawn (not fork): the child must not inherit threads/finalizers from this
    # Ray actor (e.g. wandb's service thread), which deadlock a forked child.
    process = multiprocessing.get_context("spawn").Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    wait_for_server_ready(router_ip, router_port, process, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def _resolve_session_server_ports(start: int | None, workers: int) -> list[int]:
    """Return consecutive ports from an explicit start, or auto-allocate each worker port."""
    if workers < 1:
        raise ValueError("--session-server-workers must be at least 1.")
    # TODO(#1837): Refactor IP/port allocation; keep this naive for now.
    if start is None:
        search_start = random.randint(5000, 6000)
        ports = []
        while len(ports) < workers:
            port = find_available_port(search_start)
            if port not in ports:
                ports.append(port)
            search_start = port + 1
        return ports
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
    # spawn (not fork): see start_router for rationale.
    instance_ids: dict[int, str] = {}
    processes = []
    for port in ports:
        child_args = copy.copy(args)
        child_args.session_server_port = port
        child_args.session_server_instance_id = uuid.uuid4().hex
        instance_ids[port] = child_args.session_server_instance_id
        process = multiprocessing.get_context("spawn").Process(
            target=run_session_server, args=(child_args, router_url)
        )
        process.daemon = True
        process.start()
        processes.append((port, process))
    # The per-port map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for port, process in processes:
        wait_for_server_ready(ip, port, process, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Session servers launched at {ip}, ports {ports} ({len(ports)} instances)")
