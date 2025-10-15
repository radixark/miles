from typing import List, Callable

from kimina_client import AsyncKiminaClient


class LeanVerifier:
    def __init__(self):
        self._servers = _create_servers()
        self._client_cluster = _KiminaClientCluster(TODO)

    async def check(self, *args, **kwargs):
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    def __init__(self):
        self._clients = [AsyncKiminaClient(TODO) for TODO in TODO]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


# TODO handle docker stop more gracefully
class _KiminaServer:
    def __init__(self):
        self._docker_start()

    def _docker_start(self):
        TODO
