from __future__ import annotations

import asyncio

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import resolve_router_addrs, wait_router_ready, wait_session_server_ready
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class TestResolveRouterAddrs:
    async def test_records_every_models_router_on_args(self, monkeypatch):
        """The driver-visible router contract (primary ip/port, per-model map) is written exactly once, here."""
        args = make_args(sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None)

        async def _fake_wait_router_ready(model_idx: int) -> HostAndPort:
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        router_addrs = await resolve_router_addrs(args)

        assert router_addrs == {"default": HostAndPort(host="10.0.0.9", port=30000)}
        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.9", 30000)
        assert args.sglang_model_routers == {"default": ("10.0.0.9", 30000)}

    async def test_resolving_again_in_the_same_process_answers_from_the_record(self, monkeypatch):
        """The driver and an in-process controller may both resolve; the second call must not re-wait."""
        args = make_args(sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None)
        waited: list[int] = []

        async def _fake_wait_router_ready(model_idx: int) -> HostAndPort:
            waited.append(model_idx)
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        first = await resolve_router_addrs(args)
        second = await resolve_router_addrs(args)

        assert second == first
        assert waited == [0]

    async def test_an_externally_configured_router_is_rejected(self):
        """External router mode was removed, so a pre-set router address means a misconfigured run."""
        args = make_args(sglang_router_ip="10.0.0.1", sglang_router_port=3000)

        with pytest.raises(AssertionError, match="external router mode was removed"):
            await resolve_router_addrs(args)


class TestWaitRouterReady:
    async def test_returns_the_provider_addr_after_the_tcp_wait(self, monkeypatch):
        """The router address is looked up from the worker manager by the spec worker name."""
        requested: list[str] = []

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                requested.append(worker_name)
                return {"primary": HostAndPort(host="10.0.0.9", port=12345)}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready",
            lambda host, port, timeout: waited.append((host, port)),
        )

        addr = await wait_router_ready(model_idx=1, provider=_FakeProvider())

        assert requested == ["inference-router-1-0-0"]
        assert waited == [("10.0.0.9", 12345)]
        assert addr == HostAndPort(host="10.0.0.9", port=12345)


class _SessionServerProvider:
    """A backend that names its session-server cells its own way, not `pool-0`, `pool-1`."""

    cell_ids = ("session-server-east", "session-server-west")

    async def get_cell_infos(self, *, pool_id: str) -> dict[str, CellInfo]:
        return {
            cell_id: CellInfo(
                cell_id=cell_id,
                pool_id=pool_id,
                alive=True,
                worker_names=[f"{cell_id}-0"],
                workers_hash="hash",
                meta={},
            )
            for cell_id in self.cell_ids
        }


async def _sleep_instantly(seconds: float) -> None:
    return None


class TestWaitSessionServerReady:
    async def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` returns without touching any other config."""
        args = make_args(use_session_server=False)
        await wait_session_server_ready(args, provider=None)

    async def test_enabled_without_hf_checkpoint_raises(self):
        """Enabling the session server without a tokenizer source fails fast."""
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            await wait_session_server_ready(args, provider=None)

    async def test_publishes_the_manager_addrs_and_instance_ids(self, monkeypatch):
        """The driver-side contract (ip, ports, instance ids) comes from the worker manager addrs."""
        requested: list[str] = []

        class _FakeProvider(_SessionServerProvider):
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                requested.append(worker_name)
                return {"primary": HostAndPort(host="10.0.0.9", port=5004 + len(requested))}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready",
            lambda host, port, timeout: waited.append((host, port)),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            num_session_servers=2,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert requested == ["session-server-east-0", "session-server-west-0"]
        assert args.session_server_addrs == ["10.0.0.9:5005", "10.0.0.9:5006"]
        assert args.session_server_instance_ids == {
            "10.0.0.9:5005": "00112233445566aa-session-server-east",
            "10.0.0.9:5006": "00112233445566aa-session-server-west",
        }
        assert waited == [("10.0.0.9", 5005), ("10.0.0.9", 5006)]

    async def test_waits_for_a_pool_whose_cells_are_not_all_up_yet(self, monkeypatch):
        """The platform brings cells up one at a time, so the first look at the pool reports too few."""
        rounds: list[int] = []

        class _FakeProvider(_SessionServerProvider):
            async def get_cell_infos(self, *, pool_id: str) -> dict[str, CellInfo]:
                rounds.append(len(rounds))
                infos = await super().get_cell_infos(pool_id=pool_id)
                return infos if len(rounds) > 1 else {self.cell_ids[0]: infos[self.cell_ids[0]]}

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                return {"primary": HostAndPort(host="10.0.0.9", port=5005)}

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready", lambda host, port, timeout: None)
        monkeypatch.setattr(asyncio, "sleep", _sleep_instantly)

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", num_session_servers=2)
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert len(rounds) == 2

    async def test_servers_on_different_hosts_raise(self):
        """A session server pool spread across hosts violates the single-ip contract."""

        class _FakeProvider(_SessionServerProvider):
            def __init__(self):
                super().__init__()
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                self._counter += 1
                return {"primary": HostAndPort(host=f"10.0.0.{self._counter}", port=5005)}

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", num_session_servers=2)
        with pytest.raises(AssertionError):
            await wait_session_server_ready(args, provider=_FakeProvider())
