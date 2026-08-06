from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import wait_router_ready, wait_session_server_ready
from miles.utils.workers.worker_spec import HostAndPort


class TestWaitRouterReady:
    async def test_returns_the_provider_addr_after_the_tcp_wait(self, monkeypatch):
        """The router address is looked up from the worker manager by the spec worker name."""
        requested: list[str] = []

        class _FakeProvider:
            async def get_addr(self, worker_name: str) -> HostAndPort:
                requested.append(worker_name)
                return HostAndPort(host="10.0.0.9", port=12345)

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready",
            lambda host, port, timeout: waited.append((host, port)),
        )

        addr = await wait_router_ready(model_idx=1)

        assert requested == ["inference-router-1-0-0"]
        assert waited == [("10.0.0.9", 12345)]
        assert addr == HostAndPort(host="10.0.0.9", port=12345)


class TestWaitSessionServerReady:
    async def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` returns without touching any other config."""
        args = make_args(use_session_server=False)
        await wait_session_server_ready(args)

    async def test_enabled_without_hf_checkpoint_raises(self):
        """Enabling the session server without a tokenizer source fails fast."""
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            await wait_session_server_ready(args)

    async def test_publishes_the_manager_addrs_and_instance_ids(self, monkeypatch):
        """The driver-side contract (ip, ports, instance ids) comes from the worker manager addrs."""
        requested: list[str] = []

        class _FakeProvider:
            async def get_addr(self, worker_name: str) -> HostAndPort:
                requested.append(worker_name)
                return HostAndPort(host="10.0.0.9", port=5004 + len(requested))

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
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
        await wait_session_server_ready(args)

        assert requested == ["session-server-0-0", "session-server-1-0"]
        assert args.session_server_addrs == ["10.0.0.9:5005", "10.0.0.9:5006"]
        assert args.session_server_instance_ids == {
            "10.0.0.9:5005": "00112233445566aa-0",
            "10.0.0.9:5006": "00112233445566aa-1",
        }
        assert waited == [("10.0.0.9", 5005), ("10.0.0.9", 5006)]

    async def test_servers_on_different_hosts_are_each_addressed_in_full(self, monkeypatch):
        """Placement may spread the servers across hosts, so no instance may be addressed by a
        port under a host borrowed from another one."""

        class _FakeProvider:
            def __init__(self):
                self._counter = 0

            async def get_addr(self, worker_name: str) -> HostAndPort:
                self._counter += 1
                return HostAndPort(host=f"10.0.0.{self._counter}", port=5005)

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
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
        await wait_session_server_ready(args)

        assert args.session_server_addrs == ["10.0.0.1:5005", "10.0.0.2:5005"]
        assert args.session_server_instance_ids == {
            "10.0.0.1:5005": "00112233445566aa-0",
            "10.0.0.2:5005": "00112233445566aa-1",
        }
        assert waited == [("10.0.0.1", 5005), ("10.0.0.2", 5005)]
