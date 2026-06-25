import logging
import multiprocessing
import random
import uuid


from miles.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    is_port_available,
    wait_for_server_ready,
)


logger = logging.getLogger(__name__)

# Keeps multi-process session-server supervisors alive past start_session_server
# (otherwise GC would fire their atexit/teardown). One per launched server.
_SESSION_SUPERVISORS = []


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
        import copy

        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
        from miles.router.router import run_router

        router_args = copy.copy(args)
        router_args.sglang_router_ip = router_ip
        router_args.sglang_router_port = router_port

    else:
        from sglang_router.launch_router import RouterArgs

        from miles.utils.http_utils import run_router

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

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    wait_for_server_ready(router_ip, router_port, process, timeout=30)
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def start_session_server(args):
    """Start a standalone session server when ``--use-session-server`` is set.

    The session server runs as a separate process with its own port and proxies
    inference requests directly to SGLang worker engines.  It is always started
    as a standalone process regardless of whether ``--use-miles-router`` is active.
    """
    if not getattr(args, "use_session_server", False):
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    if getattr(args, "session_server_ip", None) is None:
        args.session_server_ip = args.sglang_router_ip
    if getattr(args, "session_server_port", None) is None:
        args.session_server_port = find_available_port(random.randint(5000, 6000))
    if getattr(args, "session_server_instance_id", None) is None:
        args.session_server_instance_id = uuid.uuid4().hex

    workers = getattr(args, "session_server_workers", 1)
    if workers < 1:
        raise ValueError(f"--session-server-workers must be >= 1, got {workers}")

    ip, port = args.session_server_ip, args.session_server_port
    if not is_port_available(port):
        raise RuntimeError(
            f"Port {port} is already in use — a stale session server may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

    if workers > 1:
        # Multi-process: a supervisor spawns N workers + 1 router and blocks
        # until all workers are ready, with fail-fast monitoring + no orphans.
        from miles.rollout.session.session_supervisor import SessionServerSupervisor

        supervisor = SessionServerSupervisor(args, router_url, ip, port)
        supervisor.start()
        _SESSION_SUPERVISORS.append(supervisor)  # keep alive past this call
        logger.info(f"Session server (workers={workers}) launched at {ip}:{port}")
        return supervisor

    from miles.rollout.session.session_server import run_session_server

    process = multiprocessing.Process(target=run_session_server, args=(args, router_url))
    process.daemon = True
    process.start()
    wait_for_server_ready(ip, port, process, timeout=30)
    logger.info(f"Session server launched at {ip}:{port}")
