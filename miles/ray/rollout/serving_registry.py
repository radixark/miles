from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from miles.backends.sglang_utils.sglang_api_client import WorkerType

if TYPE_CHECKING:
    from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient


class ServingRegistry(Protocol):
    async def register(self, *, worker_url: str, worker_type: WorkerType, bootstrap_port: int | None) -> None: ...

    async def unregister(self, *, worker_url: str) -> None: ...


@dataclass(frozen=True)
class SGLangServingRegistry:
    client: "SGLangRouterApiClient"

    async def register(self, *, worker_url: str, worker_type: WorkerType, bootstrap_port: int | None) -> None:
        await self.client.add_worker(
            worker_url=worker_url,
            worker_type=worker_type,
            bootstrap_port=bootstrap_port,
        )

    async def unregister(self, *, worker_url: str) -> None:
        await self.client.remove_worker(worker_url=worker_url)
