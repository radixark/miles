import subprocess
import sys

import pytest

from miles.router import router as router_module
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter, run_router


def test_miles_router_module_help_exits_successfully() -> None:
    """The Miles router module exposes its CLI help without starting the server."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "miles.router.router", "--help"],
        capture_output=True,
        text=True,
        # Generous because this imports the whole miles stack, and a loaded runner takes longer
        # than the few seconds it costs idle; the bound is here to catch a hang, not to time it.
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-json" in result.stdout


class RouterRecorder:
    def __init__(self) -> None:
        self.routers: list[MilesRouter] = []

    def create(self, config: MilesRouterConfig, verbose: bool = False) -> MilesRouter:
        router = MilesRouter(config, verbose=verbose)
        self.routers.append(router)
        return router


class UvicornRunRecorder:
    def __init__(self) -> None:
        self.apps: list[object] = []
        self.bind_addresses: list[tuple[str, int]] = []

    def run(self, app: object, *, host: str, port: int, **kwargs: object) -> None:
        self.apps.append(app)
        self.bind_addresses.append((host, port))


class TestRunRouter:
    def test_config_controls_router_construction_and_uvicorn_bind_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`run_router` builds the router from the given config and serves its app on the configured host and port."""
        config = MilesRouterConfig(
            host="192.0.2.7",
            port=21234,
            max_connections=11,
            timeout=None,
            health_check_interval=1.0,
            health_check_failure_threshold=2,
        )
        router_recorder = RouterRecorder()
        uvicorn_recorder = UvicornRunRecorder()
        monkeypatch.setattr(router_module, "configure_logger_raw", lambda name: None)
        monkeypatch.setattr(router_module.setproctitle, "setproctitle", lambda title: None)
        monkeypatch.setattr(router_module, "MilesRouter", router_recorder.create)
        monkeypatch.setattr(router_module.uvicorn, "run", uvicorn_recorder.run)

        run_router(config)

        assert len(router_recorder.routers) == 1
        assert router_recorder.routers[0].config is config
        assert uvicorn_recorder.apps == [router_recorder.routers[0].app]
        assert uvicorn_recorder.bind_addresses == [("192.0.2.7", 21234)]
