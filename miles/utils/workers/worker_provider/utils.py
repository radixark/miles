from collections.abc import Awaitable, Callable

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import RPC_PORT_NAME, NamedHostAndPorts


async def apply_cell_observation(
    *,
    cell_id: str,
    observed: CellInfo | None,
    actual_workers_hash: str | None,
    add: Callable[[str, CellInfo], Awaitable[None]],
    remove: Callable[[str], Awaitable[None]],
) -> None:
    if observed is not None and actual_workers_hash is None:
        await add(cell_id, observed)
    elif observed is None and actual_workers_hash is not None:
        await remove(cell_id)
    elif observed is not None and actual_workers_hash is not None and actual_workers_hash != observed.workers_hash:
        await remove(cell_id)
        await add(cell_id, observed)


def build_rpc_handle(*, worker_class: type, addrs: NamedHostAndPorts, pool_id: str) -> BaseWorkerHandle:
    assert RPC_PORT_NAME in addrs, f"spec {pool_id} has no {RPC_PORT_NAME!r} port to be called through"
    return RpcWorkerHandle(worker_class, server_url=addrs[RPC_PORT_NAME].addr, require_stable_boot_uuid=True)
