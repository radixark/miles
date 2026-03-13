from __future__ import annotations

from miles.utils.http_utils import MILES_HOST_IP_ENV
from tests.fast.utils.ft.testbed.train import _actor_runtime_env


def test_actor_runtime_env_sets_miles_host_ip() -> None:
    assert _actor_runtime_env() == {
        "env_vars": {
            MILES_HOST_IP_ENV: "127.0.0.1",
        }
    }
