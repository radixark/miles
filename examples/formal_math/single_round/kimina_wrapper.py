import datetime
import os
import random
from typing import List

import ray
from kimina_client import AsyncKiminaClient, CheckResponse
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.misc import exec_command, get_current_node_ip, get_free_port

# TODO handle docker stop more gracefully later
_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("MILES_KILL_PREVIOUS_KIMINA_DOCKER", "1")))


class KiminaServerAndClientCluster:
    def __init__(self):
        self._servers = _create_actor_per_node(actor_cls=_KiminaServerActor)
        self._client_cluster = _KiminaClientCluster(self._servers)

    async def check(self, *args, **kwargs) -> CheckResponse:
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    def __init__(self, servers: List["_KiminaServerActor"]):
        self._clients = [AsyncKiminaClient(api_url=ray.get(server.get_api_url.remote())) for server in servers]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_actor_per_node(actor_cls) -> List:
    # for simplicity, we use all available nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    assert len(nodes) > 0

    actors = []
    for node in nodes:
        actors.append(
            actor_cls.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                num_cpus=0.001,
            ).remote()
        )

    return actors


@ray.remote
class _KiminaServerActor:
    def __init__(self):
        self.addr = get_current_node_ip()
        self.port = get_free_port()

        if _KILL_PREVIOUS_KIMINA_DOCKER:
            _docker_stop_all()
        _docker_start(port=self.port)

    def get_api_url(self):
        return f"http://{self.addr}:{self.port}"


def _docker_start(port: int):
    name = f"kimina_lean_server_auto_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    exec_command(
        "docker run "
        "-d "
        f"--name {name} "
        "--restart unless-stopped "
        # "--env-file .env "  # do not use env yet
        f"-p 80:{port} "
        f"projectnumina/kimina-lean-server:2.0.0"
    )


def _docker_stop_all():
    exec_command(
        'ids=$(docker ps -a --filter "name=kimina_lean_server_auto" -q); '
        '[ -n "$ids" ] && docker stop $ids && docker rm $ids; '
        'true'
    )
