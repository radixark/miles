from unittest.mock import MagicMock

import pytest
import ray
import torch.distributed
from torch.distributed import TCPStore

from miles.utils.ft_utils.indep_dp import create_tcp_store

FAKE_NODE_IP = "203.0.113.7"
FAKE_STORE_PORT = 4567


class TestCreateTcpStore:
    def test_create_tcp_store_binds_a_nonblocking_master_and_advertises_its_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store must use a wildcard ephemeral bind and publish its routable address."""
        store_class = MagicMock()
        store_class.return_value.port = FAKE_STORE_PORT
        monkeypatch.setattr(torch.distributed, "TCPStore", store_class)
        monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: FAKE_NODE_IP)

        store, addr = create_tcp_store()

        store_class.assert_called_once_with(
            host_name="0.0.0.0",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        assert store is store_class.return_value
        assert addr == f"{FAKE_NODE_IP}:{FAKE_STORE_PORT}"

    def test_create_tcp_store_returns_a_master_serving_clients_on_the_advertised_port(self) -> None:
        """A remote worker dialing the advertised port must reach the master store without it waiting for workers."""
        store, addr = create_tcp_store()

        host, port = addr.rsplit(":", 1)
        client = TCPStore(host_name=host, port=int(port), is_master=False)
        store.set("indep_dp_key", "indep_dp_value")

        assert client.get("indep_dp_key") == b"indep_dp_value"
