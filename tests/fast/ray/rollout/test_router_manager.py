from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.router_manager import _launch_command_on_head, start_session_server, wait_router_ready
from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.ports import resolve_session_server_ports
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.worker_spec import HostAndPort


def _recording_probe(waited: list[tuple[str, int]]) -> Callable[..., Coroutine[Any, Any, None]]:
    async def _probe(host: str, port: int, timeout: float) -> None:
        waited.append((host, port))

    return _probe


class TestLaunchCommandOnHead:
    def test_runs_the_command_on_a_head_command_actor(self, monkeypatch):
        """The command runs inside a head-pinned CommandActor, not a driver subprocess."""
        captured: dict = {}

        def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("miles.ray.rollout.router_manager.create_head_worker_actor", _create)

        actor_handle = _launch_command_on_head("python -m x --flag 'a b'")

        assert captured["worker_cls"] is CommandActor
        assert captured["env_vars"] == {}
        actor_handle.run.remote.assert_called_once_with(cmd="python -m x --flag 'a b'", envs={})


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
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        addr = await wait_router_ready(model_idx=1)

        assert requested == ["inference-router-1-0-0"]
        assert waited == [("10.0.0.9", 12345)]
        assert addr == HostAndPort(host="10.0.0.9", port=12345)


class TestStartSessionServerLaunchCommand:
    def test_one_launch_per_port_with_parseable_configs(self, monkeypatch):
        """Each resolved port gets its own subprocess with a lossless config."""
        launches: list[list[str]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager._launch_command_on_head",
            lambda launch_cmd: launches.append(shlex.split(launch_cmd)) or MagicMock(),
        )
        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready", lambda *fn_args, **fn_kwargs: None)
        monkeypatch.setattr("miles.ray.rollout.router_manager.is_port_available", lambda port: True)

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=3000,
            session_server_port=[5005, 5007],
            miles_router_timeout=None,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            tito_allowed_append_roles=["tool"],
            use_rollout_indexer_replay=False,
        )
        start_session_server(args)

        assert len(launches) == 2
        configs = []
        for argv in launches:
            assert argv[:3] == [sys.executable, "-m", "miles.rollout.session.server"]
            configs.append(parse_config_argv(SessionServerConfig, argv[3:]))

        assert {config.backend_url for config in configs} == {"http://127.0.0.1:3000"}
        assert {config.port for config in configs} == {5005, 5006}
        assert {config.host for config in configs} == {"127.0.0.1"}
        assert {config.instance_id for config in configs} == set(args.session_server_instance_ids.values())

    def test_instance_ids_are_the_run_uuid_plus_the_instance_index(self, monkeypatch):
        """Ids stay unique across runs through the run uuid and unique within one through the index."""
        monkeypatch.setattr("miles.ray.rollout.router_manager._launch_command_on_head", lambda launch_cmd: MagicMock())
        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready", lambda *fn_args, **fn_kwargs: None)
        monkeypatch.setattr("miles.ray.rollout.router_manager.is_port_available", lambda port: True)

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=3000,
            session_server_port=[5005, 5008],
            run_uuid="00112233445566aa",
            miles_router_timeout=None,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            tito_allowed_append_roles=["tool"],
            use_rollout_indexer_replay=False,
        )
        start_session_server(args)

        assert args.session_server_instance_ids == {
            5005: "00112233445566aa-0",
            5006: "00112233445566aa-1",
            5007: "00112233445566aa-2",
        }


class TestStartSessionServer:
    def test_disabled_returns_silently(self):
        """Happy no-op: ``use_session_server=False`` → return without raising,
        without touching any other config."""
        args = make_args(use_session_server=False)
        start_session_server(args)

    def test_enabled_without_hf_checkpoint_raises(self):
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            start_session_server(args)

    @pytest.mark.parametrize("workers", [0, -1])
    def test_a_non_positive_worker_count_is_rejected(self, workers):
        """A zero count published an empty port list, so the run only failed once a session was requested."""
        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=workers)
        with pytest.raises(ValueError, match="session-server-workers"):
            start_session_server(args)

    def test_enabled_port_conflict_raises_runtime_error(self):
        """When a configured ``session_server_port`` is already bound, fail
        loud rather than silently re-using the stale process."""
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            sglang_router_ip="127.0.0.1",
            sglang_router_port=20000,
            session_server_ip="127.0.0.1",
            session_server_port=20001,
        )
        with patch("miles.ray.rollout.router_manager.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="already in use"):
                start_session_server(args)


class TestResolveSessionServerPorts:
    def test_none_auto_allocates_one_port(self):
        with patch("miles.ray.rollout.router_manager.find_available_port", return_value=20002):
            assert resolve_session_server_ports(None) == [20002]

    def test_single_value_is_a_single_server(self):
        assert resolve_session_server_ports([30000]) == [30000]

    def test_two_values_expand_to_half_open_range(self):
        assert resolve_session_server_ports([30000, 30004]) == [30000, 30001, 30002, 30003]

    def test_empty_range_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resolve_session_server_ports([30004, 30000])

    def test_more_than_two_values_raises(self):
        with pytest.raises(ValueError, match="one port or a start/end range"):
            resolve_session_server_ports([30000, 30001, 30002])
