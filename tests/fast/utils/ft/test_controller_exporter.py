from __future__ import annotations

from prometheus_client import CollectorRegistry

from miles.utils.ft.controller.controller_exporter import ControllerExporter
from tests.fast.utils.ft.conftest import get_sample_value


class TestControllerExporterGauges:
    def test_initial_mode_is_zero(self) -> None:
        registry = CollectorRegistry()
        ControllerExporter(registry=registry)
        assert get_sample_value(registry, "ft_controller_mode") == 0.0

    def test_update_mode(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_mode(1)

        assert get_sample_value(registry, "ft_controller_mode") == 1.0

    def test_update_tick_count_increments(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_tick_count()
        exporter.update_tick_count()
        exporter.update_tick_count()

        assert get_sample_value(registry, "ft_controller_tick_count_total") == 3.0

    def test_update_evicted_node_count(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_evicted_node_count(5)

        assert get_sample_value(registry, "ft_controller_evicted_node_count") == 5.0

    def test_update_recovery_phase(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_recovery_phase(2)

        assert get_sample_value(registry, "ft_controller_recovery_phase") == 2.0

    def test_update_training_job_status(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_training_job_status(-1)

        assert get_sample_value(registry, "ft_training_job_status") == -1.0

    def test_update_training_metrics_loss_and_mfu(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)

        exporter.update_training_metrics(loss=2.5, mfu=0.42)

        assert get_sample_value(registry, "ft_training_loss_latest") == 2.5
        assert get_sample_value(registry, "ft_training_mfu_latest") == 0.42

    def test_update_training_metrics_none_values_no_change(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        exporter.update_training_metrics(loss=1.0, mfu=0.5)

        exporter.update_training_metrics(loss=None, mfu=None)

        assert get_sample_value(registry, "ft_training_loss_latest") == 1.0
        assert get_sample_value(registry, "ft_training_mfu_latest") == 0.5


class TestControllerExporterAddress:
    def test_default_port(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(registry=registry)
        assert exporter.address == "http://localhost:9400"

    def test_custom_port(self) -> None:
        registry = CollectorRegistry()
        exporter = ControllerExporter(port=9500, registry=registry)
        assert exporter.address == "http://localhost:9500"
