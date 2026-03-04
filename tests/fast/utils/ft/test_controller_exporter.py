from __future__ import annotations

from prometheus_client import CollectorRegistry

import miles.utils.ft.metric_names as mn
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from tests.fast.utils.ft.conftest import get_sample_value


class TestControllerExporterGauges:
    def test_initial_mode_is_zero(self) -> None:
        registry = CollectorRegistry()
        ControllerExporter(registry=registry)
        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 0.0

    def test_update_mode(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_mode(1)

        assert get_sample_value(registry, mn.CONTROLLER_MODE) == 1.0

    def test_update_tick_count_increments(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_tick_count()
        exporter.update_tick_count()
        exporter.update_tick_count()

        assert get_sample_value(registry, mn.CONTROLLER_TICK_COUNT + "_total") == 3.0

    def test_update_evicted_node_count(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_evicted_node_count(5)

        assert get_sample_value(registry, mn.CONTROLLER_EVICTED_NODE_COUNT) == 5.0

    def test_update_recovery_phase(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_recovery_phase(2)

        assert get_sample_value(registry, mn.CONTROLLER_RECOVERY_PHASE) == 2.0

    def test_update_training_job_status(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_training_job_status(-1)

        assert get_sample_value(registry, mn.TRAINING_JOB_STATUS) == -1.0

    def test_update_training_metrics_loss_and_mfu(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_training_metrics(loss=2.5, mfu=0.42)

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 2.5
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.42

    def test_update_training_metrics_none_values_no_change(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        exporter.update_training_metrics(loss=1.0, mfu=0.5)

        exporter.update_training_metrics(loss=None, mfu=None)

        assert get_sample_value(registry, mn.TRAINING_LOSS_LATEST) == 1.0
        assert get_sample_value(registry, mn.TRAINING_MFU_LATEST) == 0.5


class TestControllerExporterAddress:
    def test_default_port(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        assert exporter.address == "http://localhost:9400"

    def test_custom_port(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(port=9500, registry=registry)
        assert exporter.address == "http://localhost:9500"


class TestControllerExporterLifecycle:
    def test_stop_shuts_down_http_server(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        exporter._port = port
        exporter.start()
        assert exporter._httpd is not None

        exporter.stop()
        assert exporter._httpd is None

    def test_stop_idempotent_when_not_started(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        exporter.stop()
        assert exporter._httpd is None
