"""Unit tests for TrainingRankExporter.

TrainingRankExporter owns the Prometheus exporter and metric gauges
(heartbeat + phase). These tests verify gauge creation, updates, and the
HTTP exposition endpoint.
"""

from collections.abc import Iterator

import httpx
import pytest

from miles.utils.ft.agents.metrics.training_rank_exporter import TrainingRankExporter


def _parse_gauge(text: str, metric_name: str, labels: dict[str, str]) -> float:
    """Extract a gauge value from Prometheus text exposition format."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if metric_name not in line:
            continue
        label_match = all(f'{k}="{v}"' in line for k, v in labels.items())
        if label_match:
            value_str = line.rsplit(" ", 1)[-1]
            return float(value_str)
    raise ValueError(f"{metric_name} with labels {labels} not found in metrics output")


@pytest.fixture()
def metric_exporter() -> Iterator[TrainingRankExporter]:
    exporter = TrainingRankExporter(rank=0, node_id="test-node")
    yield exporter
    exporter.shutdown()


class TestTrainingRankExporterExporter:
    @pytest.mark.anyio
    async def test_exporter_returns_prometheus_format(self, metric_exporter: TrainingRankExporter) -> None:
        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.anyio
    async def test_exporter_address_has_port(self, metric_exporter: TrainingRankExporter) -> None:
        address = metric_exporter.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.anyio
    async def test_initial_gauge_values(self, metric_exporter: TrainingRankExporter) -> None:
        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "miles_ft_agent_heartbeat" in text
        assert "miles_ft_training_phase" in text
        assert 'rank="0"' in text


class TestTrainingRankExporterStep:
    @pytest.mark.anyio
    async def test_step_increments_heartbeat(self, metric_exporter: TrainingRankExporter) -> None:
        metric_exporter.step()
        metric_exporter.step()
        metric_exporter.step()

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        heartbeat = _parse_gauge(response.text, "miles_ft_agent_heartbeat", labels)
        assert heartbeat == 3.0

    @pytest.mark.anyio
    async def test_step_heartbeat_monotonic_across_phases(self, metric_exporter: TrainingRankExporter) -> None:
        """Simulate a full rollout cycle with split set_phase/step API."""
        address = metric_exporter.get_exporter_address()
        labels = {"rank": "0"}

        # set_phase bumps heartbeat too
        metric_exporter.set_phase("training")
        for _ in range(4):
            metric_exporter.step()

        metric_exporter.set_phase("idle")
        metric_exporter.set_phase("checkpoint_saving")
        metric_exporter.set_phase("idle")

        metric_exporter.set_phase("training")
        for _ in range(4):
            metric_exporter.step()

        metric_exporter.set_phase("idle")

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{address}/metrics")

        # 6 set_phase calls + 8 step calls = 14 heartbeat bumps
        heartbeat = _parse_gauge(resp.text, "miles_ft_agent_heartbeat", labels)
        assert heartbeat == 14.0
        phase = _parse_gauge(resp.text, "miles_ft_training_phase", labels)
        assert phase == 0.0

    def test_step_exception_does_not_propagate(self, metric_exporter: TrainingRankExporter) -> None:
        from unittest.mock import patch

        with patch.object(metric_exporter, "_heartbeat_child", **{"set.side_effect": RuntimeError("boom")}):
            metric_exporter.step()


class TestTrainingRankExporterSetPhase:
    @pytest.mark.anyio
    async def test_set_phase_updates_phase_gauge(self, metric_exporter: TrainingRankExporter) -> None:
        metric_exporter.set_phase("checkpoint_saving")

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert phase == 2.0

    @pytest.mark.anyio
    async def test_set_phase_also_bumps_heartbeat(self, metric_exporter: TrainingRankExporter) -> None:
        metric_exporter.set_phase("training")
        metric_exporter.set_phase("idle")

        address = metric_exporter.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        labels = {"rank": "0"}
        heartbeat = _parse_gauge(response.text, "miles_ft_agent_heartbeat", labels)
        assert heartbeat == 2.0
        phase = _parse_gauge(response.text, "miles_ft_training_phase", labels)
        assert phase == 0.0
