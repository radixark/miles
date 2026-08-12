from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata, compute_nodes_per_engine
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


def _make_meta(**overrides) -> ServerCellMetadata:
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": "inference-engine-0-0-0",
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": "inference-engine-0-0-0-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


class _StubProvider:
    def __init__(self, addrs: dict[str, HostAndPort]):
        self._addrs = addrs
        self.requested_worker_names: list[str] = []

    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        self.requested_worker_names.append(worker_name)
        return self._addrs


@pytest.fixture
def stub_provider():
    def _install(addrs: dict[str, HostAndPort]) -> _StubProvider:
        return _StubProvider(addrs)

    return _install


def _make_cell(provider: _StubProvider, **meta_overrides) -> ServerCell:
    return track_server_cell(
        ServerCell(args=make_args(), meta=_make_meta(**meta_overrides), router_api_client=None, provider=provider)
    )


class TestComputeAddrInfo:
    async def test_the_cell_addresses_are_read_from_its_rank_zero_worker(self, stub_provider):
        """A cell is addressed through the worker holding its master-mode ports."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert provider.requested_worker_names == ["inference-engine-0-0-0-0"]
        assert addr_info.server_url == "http://10.0.0.1:30000"

    async def test_the_gate_url_points_at_the_out_of_band_control_port(self, stub_provider):
        """Activation must target the gate port, never the engine's serving port."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13007),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.gate_url == "http://10.0.0.1:13007"

    async def test_a_prefill_cell_also_carries_its_disaggregation_bootstrap_port(self, stub_provider):
        """PD disaggregation needs this port published to the router alongside the url."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
                disaggregation_bootstrap=HostAndPort(host="10.0.0.1", port=11000),
            )
        )

        addr_info = await _make_cell(provider, worker_type="prefill")._compute_addr_info()

        assert addr_info.bootstrap_port == 11000

    async def test_a_regular_cell_has_no_bootstrap_port(self, stub_provider):
        """Only prefill engines allocate a bootstrap port, so the rest must report None."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.bootstrap_port is None

    async def test_ipv6_hosts_stay_bracketed_in_both_urls(self, stub_provider):
        """Unbracketed ipv6 literals would make both urls unparseable."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="[fd00::1]", port=30000),
                gate=HostAndPort(host="[fd00::1]", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.server_url == "http://[fd00::1]:30000"
        assert addr_info.gate_url == "http://[fd00::1]:13000"


class TestComputeNodesPerEngine:
    def test_an_engine_that_fits_inside_one_node_still_occupies_a_whole_node(self):
        """Plain integer division would report zero nodes for every engine smaller than a node."""
        assert compute_nodes_per_engine(num_gpus_per_engine=1, num_gpus_per_node=8) == 1
        assert compute_nodes_per_engine(num_gpus_per_engine=8, num_gpus_per_node=8) == 1

    def test_an_engine_spanning_several_nodes_reports_how_many_it_spans(self):
        """A multi-node engine is launched once per node it covers, so the count must scale with it."""
        assert compute_nodes_per_engine(num_gpus_per_engine=32, num_gpus_per_node=8) == 4


class TestTheApiClientOfACell:
    async def test_carries_the_engine_api_key(self, stub_provider) -> None:
        """A protected /server_info answers 401 without it, so the engine env would never be recorded."""
        cell = _make_cell(
            stub_provider(dict(primary=HostAndPort(host="10.0.0.1", port=30000))), sglang_api_key="a-key"
        )
        await cell.init()

        assert cell.api_client.api_key == "a-key"
