from dataclasses import dataclass

import pytest
import uvicorn

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.utils.ft_utils.api_server import server
from miles.utils.workers.worker_provider.base import CellInfo


@dataclass
class _CellApi:
    operations: "_CellOperations"
    controller: "_Controller"
    servers: list["_Server"]


@pytest.fixture
def cell_api(monkeypatch: pytest.MonkeyPatch) -> _CellApi:
    environment = _CellApi(operations=_CellOperations(), controller=_Controller(), servers=[])

    def create_server(config: uvicorn.Config) -> _Server:
        instance = _Server(config=config)
        environment.servers.append(instance)
        return instance

    monkeypatch.setattr(server.uvicorn, "Server", create_server)
    return environment


class _Server:
    def __init__(self, *, config: uvicorn.Config) -> None:
        self.config = config
        self.started = False
        self.should_exit = False

    def run(self) -> None:
        self.started = True


class _Controller:
    async def get_cell_statuses(self) -> dict:
        return {}


class _CellOperations:
    async def cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        infos = [
            _info(cell_id="new-rollout-cell", pool_id="new-rollout", category=POOL_CATEGORY_INFERENCE_ENGINE),
            _info(cell_id="trainer-cell", pool_id="trainer-engine-actor", category="trainer_engine"),
            _info(cell_id="session-cell", pool_id="session", category=None),
        ]
        return {
            info.cell_id: info
            for info in infos
            if (pool_ids is None or info.pool_id in pool_ids)
            and (category is None or info.meta["category"] == category)
        }


def _info(*, cell_id: str, pool_id: str, category: str | None) -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=True,
        worker_names=[],
        workers_hash="hash",
        meta={"category": category},
    )
