import asyncio
import logging
from collections.abc import Sequence

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs.inference import (
    compute_router_worker_name,
    compute_session_server_instance_id,
    session_server_worker_name,
)
from miles.utils.http_utils import wait_tcp_ready
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

_ROUTER_READY_TIMEOUT_SECONDS = 30.0
_SESSION_SERVER_READY_TIMEOUT_SECONDS = 300.0


async def resolve_router_addrs(args, *, router_providers: Sequence[BaseWorkerProvider]) -> dict[str, HostAndPort]:
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
    assert len(router_providers) == len(config.models), (
        f"every model is served by its own router, so it needs its own provider "
        f"(got {len(router_providers)} for {len(config.models)} models)"
    )
    ready = await asyncio.gather(
        *[
            wait_router_ready(model_idx=model_idx, provider=router_providers[model_idx])
            for model_idx in range(len(config.models))
        ]
    )
    router_addrs = {model_cfg.name: addr for model_cfg, addr in zip(config.models, ready, strict=True)}

    primary = router_addrs[config.models[0].name]
    args.sglang_router_ip = primary.host
    args.sglang_router_port = primary.port
    args.sglang_model_routers = {name: (addr.host, addr.port) for name, addr in router_addrs.items()}

    return router_addrs


async def wait_router_ready(*, model_idx: int, provider: BaseWorkerProvider) -> HostAndPort:
    """Wait until the model's router, launched by the platform, is reachable and return its address."""
    worker_name = compute_router_worker_name(model_idx)
    router_addr = (await provider.get_addrs(worker_name=worker_name))["primary"]
    await wait_tcp_ready(router_addr.host, router_addr.port, timeout=_ROUTER_READY_TIMEOUT_SECONDS)
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

    assert provider is not None
    addrs = [
        named["primary"]
        for named in await asyncio.gather(
            *[
                provider.get_addrs(worker_name=session_server_worker_name(index))
                for index in range(args.num_session_servers)
            ]
        )
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
    await asyncio.gather(
        *[wait_tcp_ready(addr.host, addr.port, timeout=_SESSION_SERVER_READY_TIMEOUT_SECONDS) for addr in addrs]
    )
    logger.info(f"Session servers ready at {args.session_server_addrs} ({len(addrs)} instances)")
