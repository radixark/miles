import os
from typing import List, Callable

from kimina_client import AsyncKiminaClient, CheckResponse

_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("MILES_KILL_PREVIOUS_KIMINA_DOCKER", "1")))


class KiminaServerAndClientCluster:
    def __init__(self):
        self._servers = _create_servers()
        self._client_cluster = _KiminaClientCluster(self._servers)

    async def check(self, *args, **kwargs) -> CheckResponse:
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    def __init__(self, servers: List["_KiminaServer"]):
        self._clients = [AsyncKiminaClient(api_url=server.api_url) for server in servers]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_servers() -> List["_KiminaServer"]:
    return TODO


# TODO handle docker stop more gracefully
class _KiminaServer:
    def __init__(self):
        if _KILL_PREVIOUS_KIMINA_DOCKER:
            _docker_stop_all()
        _docker_start()

    @property
    def api_url(self):
        return TODO


def _docker_start():
    TODO


def _docker_stop_all():
    TODO
