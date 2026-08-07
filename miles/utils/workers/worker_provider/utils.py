from collections.abc import Awaitable, Callable

from miles.utils.workers.worker_provider.base import CellInfo


def single_worker_name_of(cell_infos: dict[str, CellInfo], *, pool_id: str) -> str:
    assert (
        len(cell_infos) == 1
    ), f"pool {pool_id} is addressed as a single-cell pool, but the backend reports {sorted(cell_infos)}"
    (info,) = cell_infos.values()
    assert (
        len(info.worker_names) == 1
    ), f"cell {info.cell_id} is addressed as a single-worker cell, but it runs {info.worker_names}"
    return info.worker_names[0]


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
