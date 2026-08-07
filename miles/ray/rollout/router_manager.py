import asyncio
import logging
import time

from miles.ray.specs.inference import compute_router_spec_name, compute_session_server_instance_id
from miles.utils.http_utils import is_tcp_ready
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

WAIT_SERVING_TIMEOUT_SECONDS = 30.0
WAIT_SERVING_POLL_INTERVAL_SECONDS = 0.5


async def wait_router_ready(model_idx: int) -> HostAndPort:
    """Wait until the model's router, launched by the RayWorkerManager, is reachable and return its address."""
    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    worker_name = compute_worker_name(spec_name=compute_router_spec_name(model_idx))
    router_addr = await provider.get_addr(worker_name=worker_name)
    await wait_worker_serving(provider=provider, worker_name=worker_name, addr=router_addr)
    logger.info(f"Router ready at {router_addr}")
    return router_addr


async def wait_worker_serving(
    *,
    provider: BaseWorkerProvider,
    worker_name: str,
    addr: HostAndPort,
    timeout: float = WAIT_SERVING_TIMEOUT_SECONDS,
) -> None:
    """Wait for a launched worker's port, giving up early once the worker itself is gone.

    A server that dies at import time -- a bad template path, a missing model -- never opens the
    port, so waiting the whole timeout reports a network problem for what is a crashed child whose
    traceback is already in its own log.
    """
    deadline = time.monotonic() + timeout
    while True:
        if is_tcp_ready(addr.host, addr.port):
            return
        if not await provider.is_worker_alive(worker_name):
            raise RuntimeError(
                f"Worker {worker_name} died before {addr.host}:{addr.port} accepted connections; "
                f"its own log holds the reason it exited"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Worker {worker_name} is alive but {addr.host}:{addr.port} is not ready after {timeout}s"
            )
        await asyncio.sleep(WAIT_SERVING_POLL_INTERVAL_SECONDS)


async def start_session_server(args):
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

    provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
    addrs = [
        await provider.get_addr(worker_name=compute_worker_name(spec_name="session-server", cell_index=i))
        for i in range(args.num_session_servers)
    ]
    # The canonical driver-side value; rollout code picks from this list. Instances may sit on
    # different hosts, so each one is addressed in full rather than by a port under a shared ip.
    args.session_server_addrs = [f"{x.host}:{x.port}" for x in addrs]

    # Spawn all children before waiting on any: each child pays the ~10s
    # transformers import, so N servers start in ~one import of wall-time.
    instance_ids: dict[str, str] = {}
    for instance_index, addr in enumerate(args.session_server_addrs):
        instance_ids[addr] = compute_session_server_instance_id(args, instance_index)
    # The per-address map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for instance_index, addr in enumerate(addrs):
        await wait_worker_serving(
            provider=provider,
            worker_name=compute_worker_name(spec_name="session-server", cell_index=instance_index),
            addr=addr,
        )
    logger.info(f"Session servers launched at {args.session_server_addrs} ({len(addrs)} instances)")
