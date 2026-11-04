from pathlib import Path

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.static_launch_fixtures import (
    _launch_static_roster,
)

from miles.utils.workers.serving.utils import parse_orchestrator_argv, parse_serve_worker_config


class TestLauncherStaticConnectionTransport:
    def test_orchestrator_and_served_worker_receive_the_same_nonempty_static_roster(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One actual launch transports static router topology to both process entrypoints."""
        values = _launch_static_roster(monkeypatch=monkeypatch, directory=tmp_path)
        run = values["run"]
        orchestrator = parse_orchestrator_argv(run["orchestrator"]["command"])
        worker_command = run["trainerEngines"][0]["command"]
        worker = parse_serve_worker_config(worker_command[worker_command.index("--config") + 1])

        assert worker.static_connections == orchestrator.static_connections
        connections = orchestrator.static_connections.static_conn_infos
        assert set(connections) == {"inference-router-0"}
        router = connections["inference-router-0"]
        assert router.name == "inference-router-0"
        assert router.num_cells == router.num_workers_per_cell == router.pods_per_cell == 1
        assert router.worker_class is None
        assert [(port.name, port.static_port) for port in router.port_infos] == [("primary", 8000)]
