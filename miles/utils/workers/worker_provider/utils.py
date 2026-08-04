from collections.abc import Awaitable, Callable

from miles.utils.workers.worker_provider.base import CellInfo


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
