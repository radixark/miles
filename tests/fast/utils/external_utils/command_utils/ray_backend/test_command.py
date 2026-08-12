from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.ray_backend import command


class TestExecCommandAllRayNodes:
    def test_executes_the_substituted_command_on_each_selected_alive_node(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selected alive nodes receive substituted commands in deterministic current-node-first order."""
        nodes = [
            {"Alive": True, "NodeID": "third", "NodeManagerAddress": "10.0.0.3"},
            {"Alive": False, "NodeID": "dead", "NodeManagerAddress": "10.0.0.0"},
            {"Alive": True, "NodeID": "current", "NodeManagerAddress": "10.0.0.2"},
            {"Alive": True, "NodeID": "first", "NodeManagerAddress": "10.0.0.1"},
        ]
        fake_ray, ray_execution = fake_ray_factory(nodes)
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.2")
        monkeypatch.setattr(
            command,
            "run_shell_command",
            lambda cmd, *, capture_output: f"{capture_output}: {cmd}",
        )

        results = command.exec_command_all_ray_nodes(
            "rank={{node_rank}} count={{nnodes}} master={{master_addr}} node={{node_ip}}",
            capture_output=True,
            num_nodes=2,
        )

        assert fake_ray.init_addresses == ["auto"]
        assert ray_execution.scheduled_node_ids == ["current", "first"]
        assert results == [
            "True: unset CUDA_VISIBLE_DEVICES; rank=0 count=2 master=10.0.0.2 node=10.0.0.2",
            "True: unset CUDA_VISIBLE_DEVICES; rank=1 count=2 master=10.0.0.2 node=10.0.0.1",
        ]
        assert fake_ray.shutdown_count == 1

    @pytest.mark.parametrize(
        ("nodes", "num_nodes", "message"),
        [
            ([{"Alive": False, "NodeID": "dead", "NodeManagerAddress": "10.0.0.0"}], None, None),
            ([{"Alive": True, "NodeID": "only", "NodeManagerAddress": "10.0.0.1"}], 2, "only 1 alive"),
        ],
    )
    def test_rejects_an_unschedulable_node_selection_and_shuts_ray_down(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
        nodes: list[dict[str, object]],
        num_nodes: int | None,
        message: str | None,
    ) -> None:
        """An impossible node selection raises its assertion and always shuts Ray down."""
        fake_ray, _ = fake_ray_factory(nodes)
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.1")

        with pytest.raises(AssertionError, match=message):
            command.exec_command_all_ray_nodes("true", num_nodes=num_nodes)

        assert fake_ray.shutdown_count == 1

    def test_shuts_ray_down_when_collecting_remote_results_fails(
        self,
        fake_ray_factory: Callable[..., tuple[Any, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A result-collection failure propagates after Ray is shut down exactly once."""
        nodes = [{"Alive": True, "NodeID": "only", "NodeManagerAddress": "10.0.0.1"}]
        fake_ray, _ = fake_ray_factory(nodes, get_error=RuntimeError("remote failed"))
        monkeypatch.setattr(command, "get_current_node_ip", lambda: "10.0.0.1")

        with pytest.raises(RuntimeError, match="remote failed"):
            command.exec_command_all_ray_nodes("true")

        assert fake_ray.shutdown_count == 1


class TestStartMooncakeMaster:
    def test_reports_an_unreadable_log_when_mooncake_startup_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Failed startup cleanup reports when the nonexistent diagnostic log cannot be read."""
        commands: list[str] = []
        log_path = tmp_path / "missing" / "mooncake.log"
        monkeypatch.setattr(command, "_is_tcp_server_ready", lambda host, port: False)
        monkeypatch.setattr(command, "run_shell_command", commands.append)

        def fail_wait(host: str, port: int, *, timeout: float) -> None:
            raise RuntimeError("not ready")

        monkeypatch.setattr(command, "wait_for_server_ready", fail_wait)

        with pytest.raises(RuntimeError, match=r"unable to read .*mooncake\.log"):
            command.start_mooncake_master(log_path=log_path)

        assert len(commands) == 2
        assert "pkill -x mooncake_master" in commands[1]
