import logging

from miles.ray.specs.inference import compute_router_pool_id, compute_session_server_instance_id
from miles.rollout.session.types import SessionServerInstance
from miles.utils.http_utils import wait_tcp_ready_async
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


async def wait_router_ready(model_idx: int) -> HostAndPort:
    """Wait until the model's router, launched by the RayWorkerManager, is reachable and return its address."""
    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    worker_name = compute_worker_name(pool_id=compute_router_pool_id(model_idx))
    router_addr = (await provider.get_addrs(worker_name=worker_name))["primary"]
    await wait_tcp_ready_async(router_addr.host, router_addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router ready at {router_addr}")
    return router_addr


async def wait_session_server_ready(args):
    """Start the standalone session servers when ``--use-session-server`` is set.

    One independent single-process server per resolved port; the rollout side
    picks one per session and its URL carries the affinity from then on.
    Always runs standalone regardless of whether ``--use-miles-router`` is
    active.
    """
    if not getattr(args, "use_session_server", False):
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    if args.session_server_workers < 1:
        raise ValueError("--session-server-workers must be at least 1.")

    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    addrs = [
        (await provider.get_addrs(worker_name=compute_worker_name(pool_id="session-server", cell_index=i)))["primary"]
        for i in range(args.session_server_workers)
    ]
    # The canonical driver-side value; rollout code picks one instance from this
    # list per session. Each record carries the full address -- instances may sit
    # on different hosts -- and the instance id that replaced the per-session
    # /health probe, so a pick never has to line up with any other structure.
    args.session_server_instances = [
        SessionServerInstance(
            addr=f"{x.host}:{x.port}",
            instance_id=compute_session_server_instance_id(args, instance_index),
        )
        for instance_index, x in enumerate(addrs)
    ]

    for addr in addrs:
        await wait_tcp_ready_async(addr.host, addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Session servers ready at {[x.addr for x in args.session_server_instances]} ({len(addrs)} instances)")
