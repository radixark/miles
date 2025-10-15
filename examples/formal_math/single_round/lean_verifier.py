from typing import List

from kimina_client import AsyncKiminaClient


class LeanVerifier:
    def __init__(self):
        self._servers = _launch_servers()
        self._clients = [AsyncKiminaClient(TODO) for server in self._servers]

    async def check(self):
        return TODO


# TODO handle docker stop more gracefully
class _KiminaServer:
    def __init__(self):
        self._docker_start()

    def _docker_start(self):
        TODO
