from __future__ import annotations

from tests.fast.utils.ft.utils.metric_injectors import get_sample_value

from miles.utils.ft.agents.core.rollout.metrics_exporter import RolloutMetricsExporter


class TestUpdate:
    def test_healthy_sets_gauge_to_one(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            exporter.update(cell_id="a0", is_healthy=True)

            assert get_sample_value(exporter.registry, "rollout_cell_alive", {"cell_id": "a0"}) == 1.0
        finally:
            exporter.shutdown()

    def test_unhealthy_sets_gauge_to_zero(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            exporter.update(cell_id="a0", is_healthy=False)

            assert get_sample_value(exporter.registry, "rollout_cell_alive", {"cell_id": "a0"}) == 0.0
        finally:
            exporter.shutdown()


class TestAddress:
    def test_address_available_immediately(self) -> None:
        exporter = RolloutMetricsExporter()

        try:
            assert exporter.address.startswith("http://")
            port = int(exporter.address.rsplit(":", 1)[-1])
            assert port > 0
        finally:
            exporter.shutdown()


class TestShutdown:
    def test_shutdown_is_safe(self) -> None:
        exporter = RolloutMetricsExporter()
        exporter.shutdown()
