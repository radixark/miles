from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

import miles.ray.rollout.router_manager as router_manager
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_router_ready, wait_session_server_ready
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_TWO_MODEL_CONFIG = """\
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 2
  - name: ref
    model_path: /fake/ref-model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 4
"""


def _make_two_model_args(tmp_path: Path) -> Namespace:
    config_path = tmp_path / "sglang_config.yaml"
    config_path.write_text(_TWO_MODEL_CONFIG)
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=12,
        num_gpus_per_node=8,
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_model_routers=None,
    )


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

    async def test_every_model_gets_its_own_router_and_model_zero_is_the_primary(self, monkeypatch, tmp_path: Path):
        """Each model has its own router, and the driver-wide ip/port must be the first model's, not the last."""
        args = _make_two_model_args(tmp_path)
        waited: list[int] = []

        async def _fake_wait_router_ready(model_idx: int) -> HostAndPort:
            waited.append(model_idx)
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        router_addrs = await resolve_router_addrs(args)

        assert waited == [0, 1]
        assert router_addrs == {
            "actor": HostAndPort(host="10.0.0.9", port=30000),
            "ref": HostAndPort(host="10.0.0.9", port=30001),
        }
        assert args.sglang_model_routers == {"actor": ("10.0.0.9", 30000), "ref": ("10.0.0.9", 30001)}
        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.9", 30000)

    async def test_an_externally_configured_router_is_rejected(self):
        """External router mode was removed, so a pre-set router address means a misconfigured run."""
        args = make_args(sglang_router_ip="10.0.0.1", sglang_router_port=3000, sglang_model_routers=None)

        with pytest.raises(AssertionError, match="external router mode was removed"):
            await resolve_router_addrs(args)


def _recording_probe(waited: list[tuple[str, int]]) -> Callable[..., Coroutine[Any, Any, None]]:
    async def _probe(host: str, port: int, timeout: float) -> None:
        waited.append((host, port))

    return _probe


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
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        addr = await wait_router_ready(model_idx=1)

        assert requested == ["inference-router-1-0-0"]
        assert waited == [("10.0.0.9", 12345)]
        assert addr == HostAndPort(host="10.0.0.9", port=12345)

    async def test_an_unreachable_router_port_fails_instead_of_returning_an_address(self, monkeypatch):
        """A router whose port never opens must fail startup rather than be reported ready."""

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                return {"primary": HostAndPort(host="10.0.0.9", port=12345)}

        async def _refuse(host: str, port: int, timeout: float) -> None:
            raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")

        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _refuse)

        with pytest.raises(RuntimeError, match="10.0.0.9:12345 not ready"):
            await wait_router_ready(model_idx=1)

    async def test_a_failed_router_addr_lookup_fails_before_any_tcp_wait(self, monkeypatch):
        """A router the worker manager cannot resolve must abort startup, not be probed anyway."""

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                raise RuntimeError("router worker is not registered")

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        with pytest.raises(RuntimeError, match="not registered"):
            await wait_router_ready(model_idx=1)
        assert waited == []


class TestWaitSessionServerReady:
    async def test_disabled_session_server_does_not_create_a_provider_or_publish_addresses(self, monkeypatch):
        """Disabling the session server publishes no addr / instance-id fields and resolves no addrs."""
        created: list[object] = []

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                raise AssertionError("the disabled branch must not resolve any addrs")

        def _create() -> _FakeProvider:
            provider = _FakeProvider()
            created.append(provider)
            return provider

        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=_create),
        )

        args = make_args(use_session_server=False)
        await wait_session_server_ready(args)

        assert created == []
        assert not hasattr(args, "session_server_addrs")
        assert not hasattr(args, "session_server_instance_ids")

    async def test_enabled_without_hf_checkpoint_raises(self):
        """Enabling the session server without a tokenizer source fails fast."""
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            await wait_session_server_ready(args)

    @pytest.mark.parametrize("workers", [0, -1])
    async def test_a_non_positive_worker_count_is_rejected(self, workers, monkeypatch):
        """A zero count published an empty address list, so the run only failed once a session was requested."""
        created: list[object] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: created.append(None)),
        )

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=workers)
        with pytest.raises(ValueError, match="session-server-workers"):
            await wait_session_server_ready(args)
        assert created == []

    async def test_publishes_the_manager_addrs_and_instance_ids(self, monkeypatch):
        """The driver-side contract (ip, ports, instance ids) comes from the worker manager addrs."""
        requested: list[str] = []

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                requested.append(worker_name)
                return {"primary": HostAndPort(host="10.0.0.9", port=5004 + len(requested))}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
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

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                self._counter += 1
                return {"primary": HostAndPort(host=f"10.0.0.{self._counter}", port=5005)}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args)

        assert args.session_server_addrs == ["10.0.0.1:5005", "10.0.0.2:5005"]
        assert args.session_server_instance_ids == {
            "10.0.0.1:5005": "00112233445566aa-0",
            "10.0.0.2:5005": "00112233445566aa-1",
        }
        assert waited == [("10.0.0.1", 5005), ("10.0.0.2", 5005)]

    async def test_one_unreachable_instance_fails_the_whole_readiness_wait(self, monkeypatch):
        """A single session server whose port never opens fails startup even if its siblings are up."""

        class _FakeProvider:
            def __init__(self):
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                self._counter += 1
                return {"primary": HostAndPort(host="10.0.0.9", port=5004 + self._counter)}

        async def _refuse_one(host: str, port: int, timeout: float) -> None:
            if port == 5006:
                raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")

        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _refuse_one)

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=2)
        with pytest.raises(RuntimeError, match="10.0.0.9:5006 not ready"):
            await wait_session_server_ready(args)

    async def test_a_failed_instance_addr_lookup_fails_before_any_tcp_wait(self, monkeypatch):
        """A session server the worker manager cannot resolve aborts startup before any TCP probe."""

        class _FakeProvider:
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                raise RuntimeError("session-server worker is not registered")

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.RayWorkerProvider",
            SimpleNamespace(create=lambda: _FakeProvider()),
        )
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=2)
        with pytest.raises(RuntimeError, match="not registered"):
            await wait_session_server_ready(args)
        assert waited == []
