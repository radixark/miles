import os
from typing import List

import ray
from kimina_client import AsyncKiminaClient, CheckResponse
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# TODO handle docker stop more gracefully later
_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("MILES_KILL_PREVIOUS_KIMINA_DOCKER", "1")))


class KiminaServerAndClientCluster:
    def __init__(self):
        self._servers = _create_servers()
        self._client_cluster = _KiminaClientCluster(self._servers)

    async def check(self, *args, **kwargs) -> CheckResponse:
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    def __init__(self, servers: List["_KiminaServerActor"]):
        self._clients = [AsyncKiminaClient(api_url=server.api_url) for server in servers]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_servers() -> List["_KiminaServerActor"]:
    # for simplicity, we use all available nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    assert len(nodes) > 0

    actors = []
    for node in nodes:
        scheduling = NodeAffinitySchedulingStrategy(node_id=(node["NodeID"]), soft=False)
        actors.append(_KiminaServerActor.options(
            name=None,
            lifetime="detached",
            scheduling_strategy=scheduling,
            num_cpus=0.001,
        ).remote())

    return actors


@ray.remote
class _KiminaServerActor:
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
