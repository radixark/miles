import logging

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
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


async def resolve_router_addrs(args) -> dict[str, HostAndPort]:
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
        model_cfg.name: await wait_router_ready(model_idx=model_idx)
        for model_idx, model_cfg in enumerate(config.models)
    }

    primary = router_addrs[config.models[0].name]
    args.sglang_router_ip = primary.host
    args.sglang_router_port = primary.port
    args.sglang_model_routers = {name: (addr.host, addr.port) for name, addr in router_addrs.items()}

    return router_addrs


async def wait_router_ready(model_idx: int) -> HostAndPort:
    """Wait until the model's router, launched by the platform, is reachable and return its address."""
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
    # OpenAIEndpointTracer.create picks each session's instance from this list.
    args.session_server_instances = [
        SessionServerInstance(
            addr=addr.netloc,
            external_addr=_compute_external_addr(args, addr),
            instance_id=compute_session_server_instance_id(args, instance_index),
        )
        for instance_index, addr in enumerate(addrs)
    ]
    _assert_hosts_keep_their_own_external_host(args.session_server_instances)

    for addr in addrs:
        await wait_tcp_ready_async(addr.host, addr.port, timeout=_SERVER_READY_TIMEOUT_SECS)
    logger.info(
        f"Session servers ready at {[instance.addr for instance in args.session_server_instances]} "
        f"({len(addrs)} instances), "
        f"externally at {[instance.external_addr for instance in args.session_server_instances]}"
    )


def _compute_external_addr(args, addr: HostAndPort) -> str:
    # spec_session_server keeps every instance on the head whenever this host is set
    if args.session_server_external_host:
        return f"{args.session_server_external_host}:{addr.port}"
    return addr.external_netloc


def _assert_hosts_keep_their_own_external_host(instances: list[SessionServerInstance]) -> None:
    placed_hosts_by_external_host: dict[str, set[str]] = {}
    for instance in instances:
        external_host = instance.external_addr.rsplit(":", 1)[0]
        placed_hosts_by_external_host.setdefault(external_host, set()).add(instance.addr.rsplit(":", 1)[0])
    for external_host, placed_hosts in placed_hosts_by_external_host.items():
        if len(placed_hosts) > 1:
            raise ValueError(
                f"Session servers on {sorted(placed_hosts)} are all published at the external host {external_host}, "
                "so agents outside the cluster would reach only one of those hosts. Set MILES_NODE_EXTERNAL_IP on "
                "each node to its own address, or pass --session-server-external-host, which keeps the session "
                "servers on the head node."
            )
