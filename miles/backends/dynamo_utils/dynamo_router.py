"""Launch a Dynamo frontend in place of ``sgl_router`` (round-robin only)."""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys

from miles.utils.http_utils import (
    _wrap_ipv6,
    find_available_port,
    get_host_info,
    is_port_available,
    wait_for_server_ready,
)

logger = logging.getLogger(__name__)


def start_dynamo_router(
    args,
    *,
    has_pd_disaggregation: bool = False,
    force_new: bool = False,
) -> tuple[str, int]:
    """Drop-in replacement for :func:`miles.ray.rollout.router_manager.start_router`
    when ``--rollout-backend dynamo`` is active.

    Same signature, same return value (``(host, port)``). Cluster-wide
    state (``args.sglang_router_ip`` / ``args.sglang_router_port``) is
    read and written the same way.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    if has_pd_disaggregation:
        raise NotImplementedError(
            "PD-disaggregation is not yet wired into the Dynamo backend; "
            "will land in a follow-up PR of this series."
        )

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if not is_port_available(router_port):
        raise RuntimeError(
            f"Port {router_port} already bound — stale Dynamo frontend? "
            f"Run 'pkill -f dynamo.frontend' and retry."
        )

    argv: list[str] = [
        sys.executable, "-m", "dynamo.frontend",
        "--http-port", str(router_port),
        "--router-mode", "round-robin",
        "--discovery-backend", "file",
    ]

    env = os.environ.copy()
    # Turn on Dynamo's RL mode on the frontend: forces token-id passthrough,
    # sets logprobs=true, etc. Without it the request schema Miles expects
    # gets rejected by the frontend's OpenAI validator.
    env["DYN_ENABLE_RL"] = "true"

    logger.info("Starting Dynamo frontend: %s", " ".join(argv))
    proc = subprocess.Popen(argv, env=env)

    # wait_for_server_ready expects .is_alive() (multiprocessing.Process API).
    class _PopenAlive:
        def __init__(self, p):
            self._p = p

        def is_alive(self):
            return self._p.poll() is None

        @property
        def pid(self):
            return self._p.pid

    healthy = False
    try:
        wait_for_server_ready(router_ip, router_port, _PopenAlive(proc), timeout=60)
        healthy = True
    finally:
        if not healthy and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    logger.info("Dynamo frontend healthy at %s:%d", router_ip, router_port)
    return router_ip, router_port
