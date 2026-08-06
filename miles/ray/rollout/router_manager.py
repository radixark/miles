import logging

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs.inference import (
    compute_router_worker_name,
    compute_session_server_instance_id,
    session_server_worker_name,
)
from miles.utils.http_utils import wait_tcp_ready_async
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

# Readiness budget for the spawned router/session-server children. The spawn
# context re-imports the heavy transformers/megatron chain (~13s typical in
# CI), and transient CI stalls have pushed startup past a 30s budget.
_SERVER_READY_TIMEOUT_SECS = 120


async def resolve_router_addrs(args, *, provider: BaseWorkerProvider) -> dict[str, HostAndPort]:
    """Wait for every model's router and record its address on ``args``, keyed by model name.

    A second call in the same process answers from the record, so the driver and an
    in-process controller may both resolve the same ``args``.
    """
    if args.sglang_router_ip is not None:
        assert args.sglang_model_routers is not None, (
            "external router mode was removed: miles always resolves its own routers "
            "(a pre-set router address without the per-model map means a misconfigured run)"
        )
        return {name: HostAndPort(host=host, port=port) for name, (host, port) in args.sglang_model_routers.items()}

    config = resolve_sglang_config(args)  # TODO avoid resolve repeatedly
    router_addrs = {
        model_cfg.name: await wait_router_ready(model_idx=model_idx, provider=provider)
        for model_idx, model_cfg in enumerate(config.models)
    }

    primary = router_addrs[config.models[0].name]
    args.sglang_router_ip = primary.host
    args.sglang_router_port = primary.port
    args.sglang_model_routers = {name: (addr.host, addr.port) for name, addr in router_addrs.items()}

    return router_addrs


async def wait_router_ready(*, model_idx: int, provider: BaseWorkerProvider) -> HostAndPort:
    """Wait until the model's router, launched by the platform, is reachable and return its address."""
    worker_name = compute_router_worker_name(model_idx)
    router_addr = (await provider.get_addrs(worker_name=worker_name))["primary"]
    await wait_tcp_ready_async(router_addr.host, router_addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Router ready at {router_addr}")
    return router_addr


async def wait_session_server_ready(args, *, provider: BaseWorkerProvider | None):
    """Wait for the standalone session servers when ``--use-session-server`` is set.

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

    assert provider is not None
    addrs = [
        (await provider.get_addrs(worker_name=session_server_worker_name(index)))["primary"]
        for index in range(args.session_server_workers)
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
    for addr in addrs:
        await wait_tcp_ready_async(addr.host, addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(f"Session servers ready at {args.session_server_addrs} ({len(addrs)} instances)")
