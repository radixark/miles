"""The rollout engines that are up, as plain data: where each one listens and which GPUs it runs on."""

from __future__ import annotations

import asyncio

from miles.utils.observers import EngineInfo, Observers
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


def alive_engine_cells(servers) -> list:
    """A multi-node engine's cell holds ``num_nodes`` actors; only the first
    (master) owns the router-visible URL. Cells that are not alive are skipped
    until recovery completes."""
    cells = []
    for server in servers.values():
        for cell in server.all_server_cells.values():
            if cell.is_pending_weights_or_serving:
                cells.append(cell)
    return cells


def worker_fingerprint(worker_infos_per_cell: list[list]) -> tuple:
    """Changes exactly when an engine worker starts, stops or moves."""
    return tuple(
        (info.name, info.generation, info.self_addrs["primary"].host, info.self_addrs["primary"].port)
        for worker_infos in worker_infos_per_cell
        for info in worker_infos
    )


async def engine_infos(cells, worker_infos_per_cell, *, provider: BaseWorkerProvider) -> list[EngineInfo]:
    flat_infos = [info for worker_infos in worker_infos_per_cell for info in worker_infos]
    handles = provider.get_handles_of_worker_infos(flat_infos)
    probed_uuids = iter(
        await asyncio.gather(
            *[
                (
                    handle._get_gpu_uuids(gpu_ids=info.gpu_ids)
                    if (handle := handles.get(info.name)) is not None
                    else _no_gpu_uuids(info)
                )
                for info in flat_infos
            ]
        )
    )

    engines = []
    for engine_rank, (cell, worker_infos) in enumerate(zip(cells, worker_infos_per_cell, strict=True)):
        worker_gpu_uuids = [next(probed_uuids) for _ in worker_infos]
        engines.append(
            EngineInfo(
                addr=cell.server_url,
                worker_type=cell.meta.worker_type.value,
                engine_rank=engine_rank,
                gpus=[
                    [info.self_addrs["primary"].host.strip("[]"), gpu_id]
                    for info in worker_infos
                    for gpu_id in info.gpu_ids
                ],
                gpu_uuids=[uuid for uuids in worker_gpu_uuids for uuid in uuids],
            )
        )
    return engines


async def _no_gpu_uuids(info) -> list[str | None]:
    return [None] * len(info.gpu_ids)


class EngineTopologyReporter:
    """Reports the engines to run observers whenever the set of engine workers changed
    (startup, fault-tolerance recovery). Steady state costs one worker-manager
    round trip and a tuple compare."""

    def __init__(self):
        self._reported_fingerprint: tuple | None = None

    async def report_if_changed(self, servers, *, provider: BaseWorkerProvider) -> None:
        observers = Observers()
        if not observers.hears("engine_topology"):
            return
        with observers.isolated("engine_topology"):
            cells = alive_engine_cells(servers)
            worker_infos_per_cell = provider.get_worker_infos(cell_ids=[cell.meta.cell_id for cell in cells])
            fingerprint = worker_fingerprint(worker_infos_per_cell)
            if fingerprint == self._reported_fingerprint:
                return
            observers.engine_topology(await engine_infos(cells, worker_infos_per_cell, provider=provider))
            self._reported_fingerprint = fingerprint
