from __future__ import annotations

import shlex
import sys
from unittest.mock import MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.rollout.router_manager import _launch_command_on_head, start_router, start_session_server
from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.ports import resolve_session_server_ports
from miles.router.config import MilesRouterConfig
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.command_actor import CommandActor


def _make_model_cfg(*worker_types: str) -> ModelConfig:
    groups = [
        ServerGroupConfig(
            worker_type=worker_type,
            num_gpus=4,
            num_gpus_per_engine=4,
            gpu_offset=0,
            needs_offload=False,
        )
        for worker_type in worker_types
    ]
    return ModelConfig(name="default", model_path=None, server_groups=groups, update_weights=True)


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


class TestStartRouter:
    def test_pd_disagg_with_miles_router_asserts(self):
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        with patch("miles.ray.rollout.router_manager.get_host_info", return_value=("h", "127.0.0.1")), patch(
            "miles.ray.rollout.router_manager.find_available_port", return_value=20000
        ):
            with pytest.raises(AssertionError, match="miles router does not support PD"):
                start_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))


class TestStartRouterLaunchCommand:
    @pytest.fixture
    def captured_launches(self, monkeypatch):
        launches: list[list[str]] = []

        def fake_launch(launch_cmd):
            launches.append(shlex.split(launch_cmd))
            return MagicMock()

        monkeypatch.setattr("miles.ray.rollout.router_manager.get_host_info", lambda: ("h", "127.0.0.1"))
        monkeypatch.setattr("miles.ray.rollout.router_manager._launch_command_on_head", fake_launch)
        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready", lambda *fn_args, **fn_kwargs: None)
        return launches

    def test_sgl_router_launches_the_native_cli(self, captured_launches, monkeypatch):
        """The sgl router runs as the upstream CLI with the allocated ports."""
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.find_available_port", lambda start: 20000 if start < 4000 else 4001
        )
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        ip, port = start_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        (argv,) = captured_launches
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "sglang_router.launch_router"]
        assert argv[argv.index("--port") + 1] == str(port) == "20000"
        assert argv[argv.index("--prometheus-port") + 1] == "4001"

    def test_miles_router_launches_with_a_parseable_config(self, captured_launches, monkeypatch):
        """The miles router command's config payload parses back losslessly."""
        monkeypatch.setattr("miles.ray.rollout.router_manager.find_available_port", lambda start: 20000)
        args = make_args(
            use_miles_router=True,
            sglang_router_ip=None,
            sglang_router_port=None,
            miles_router_max_connections=100,
            miles_router_timeout=None,
            miles_router_health_check_failure_threshold=3,
            rollout_health_check_interval=10.0,
        )
        ip, port = start_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        (argv,) = captured_launches
        assert argv[:3] == [sys.executable, "-m", "miles.router.router"]
        config = parse_config_argv(MilesRouterConfig, argv[3:])
        assert config.host == ip
        assert config.port == port == 20000
        assert config.max_connections == 100


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
        with patch("miles.rollout.session.ports.find_available_port", return_value=20002):
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
