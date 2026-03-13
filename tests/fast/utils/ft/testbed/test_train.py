from __future__ import annotations

from tests.fast.utils.ft.integration.conftest import RayNodeInfo
from tests.fast.utils.ft.testbed.train import _actor_runtime_env, _ray_node_ip_for_id

from miles.utils.http_utils import MILES_HOST_IP_ENV


def test_actor_runtime_env_sets_miles_host_ip() -> None:
    assert _actor_runtime_env(host_ip="127.0.0.4") == {
        "env_vars": {
            MILES_HOST_IP_ENV: "127.0.0.4",
        }
    }


def test_ray_node_ip_for_id_returns_matching_node_ip() -> None:
    ray_nodes = [
        RayNodeInfo(node_ip="127.0.0.1", ray_node_id="head", is_head=True),
        RayNodeInfo(node_ip="127.0.0.3", ray_node_id="worker-1", is_head=False),
    ]

    assert _ray_node_ip_for_id(ray_nodes=ray_nodes, ray_node_id="worker-1") == "127.0.0.3"
