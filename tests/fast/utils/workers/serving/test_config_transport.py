import sys
from typing import Any

import pytest
from tests.fast.utils.workers.serving.config_transport_fixtures import _ConfigTransportSpec, _ConfigTransportWorkerSpec

from miles.utils.workers.connection_config import StaticConnConfig, StaticPoolConnInfo
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.worker_config import ServeWorkerConfig
from miles.utils.workers.worker_spec import PortInfo


class TestServeWorkerConfigTransport:
    def test_startup_passes_nonempty_serialized_static_connections_to_worker_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker startup preserves the launcher connection roster before opening any server."""
        connections = StaticConnConfig(
            static_conn_infos={
                "router": StaticPoolConnInfo(
                    name="router",
                    port_infos=[PortInfo(name="http", static_port=9007)],
                    worker_class=None,
                    num_cells=2,
                    num_workers_per_cell=3,
                    pods_per_cell=1,
                )
            }
        )
        payload = ServeWorkerConfig(
            worker_type=_ConfigTransportSpec.worker_type, args={}, static_connections=connections
        )
        received: list[StaticConnConfig] = []

        def construct_worker(spec: _ConfigTransportWorkerSpec, *, static_connections: StaticConnConfig) -> Any:
            assert isinstance(spec, _ConfigTransportWorkerSpec)
            received.append(static_connections)
            raise RuntimeError("stop before opening a server")

        monkeypatch.setitem(serve_inner.SERVE_SPEC_CLASSES, _ConfigTransportSpec.worker_type, _ConfigTransportSpec)
        monkeypatch.setattr(sys, "argv", ["serve_inner", "--config", payload.model_dump_json()])
        monkeypatch.setattr(serve_inner, "create_worker", construct_worker)

        with pytest.raises(RuntimeError, match="stop before opening a server"):
            serve_inner.main()

        assert received == [connections]
        assert received[0] is not connections
