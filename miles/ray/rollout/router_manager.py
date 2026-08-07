import logging

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.specs.inference import (
    SESSION_SERVER_POOL_ID,
    compute_router_worker_name,
    compute_session_server_instance_id,
)
from miles.utils.http_utils import wait_tcp_ready
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)


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
    wait_tcp_ready(router_addr.host, router_addr.port, timeout=30)
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
    cell_infos = await provider.cell_infos(pool_id=SESSION_SERVER_POOL_ID)
    cell_ids = sorted(cell_infos)
    assert len(cell_ids) == args.num_session_servers, (
        f"--num-session-servers asks for {args.num_session_servers} session servers but the backend reports "
        f"{len(cell_ids)}: {cell_ids}"
    )
    worker_names = [cell_infos[cell_id].worker_names[0] for cell_id in cell_ids]
    addrs = [(await provider.get_addrs(worker_name=worker_name))["primary"] for worker_name in worker_names]
    # The canonical driver-side value; rollout code picks from this list. Instances may sit on
    # different hosts, so each one is addressed in full rather than by a port under a shared ip.
    args.session_server_addrs = [f"{x.host}:{x.port}" for x in addrs]

    # Spawn all children before waiting on any: each child pays the ~10s
    # transformers import, so N servers start in ~one import of wall-time.
    instance_ids: dict[str, str] = {
        addr: compute_session_server_instance_id(args, cell_id)
        for addr, cell_id in zip(args.session_server_addrs, cell_ids, strict=True)
    }
    # The per-address map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = instance_ids
    for addr in addrs:
        wait_tcp_ready(addr.host, addr.port, timeout=30)
    logger.info(f"Session servers ready at {args.session_server_addrs} ({len(addrs)} instances)")
