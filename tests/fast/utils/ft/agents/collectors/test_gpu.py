from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from tests.fast.utils.ft.utils import make_mock_pynvml

from miles.utils.ft.agents.collectors.gpu import GpuCollector


@contextmanager
def _patched_gpu_collector(
    mock_pynvml: MagicMock | None = None, **pynvml_kwargs: Any
) -> Iterator[tuple[GpuCollector, MagicMock]]:
    if mock_pynvml is None:
        mock_pynvml = make_mock_pynvml(**pynvml_kwargs)
    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        yield GpuCollector(), mock_pynvml


class TestGpuCollector:
    async def test_normal_8_gpus_produces_48_metrics(self) -> None:
        with _patched_gpu_collector(device_count=8) as (collector, _):
            result = await collector.collect()

        assert len(result.metrics) == 48  # 8 GPUs × 6 metrics

        gpu0_names = {m.name for m in result.metrics if m.labels.get("gpu") == "0"}
        assert gpu0_names == {
            "miles_ft_gpu_available",
            "miles_ft_dcgm_fi_dev_gpu_temp",
            "miles_ft_dcgm_fi_dev_row_remap_pending",
            "miles_ft_dcgm_fi_dev_row_remap_failure",
            "miles_ft_dcgm_fi_dev_pcie_tx_throughput",
            "miles_ft_dcgm_fi_dev_gpu_util",
        }

    async def test_failing_handle_reports_gpu_unavailable(self) -> None:
        with _patched_gpu_collector(device_count=4, failing_handle_indices={2}) as (collector, _):
            result = await collector.collect()

        gpu2_metrics = [m for m in result.metrics if m.labels.get("gpu") == "2"]
        assert len(gpu2_metrics) == 1
        assert gpu2_metrics[0].name == "miles_ft_gpu_available"
        assert gpu2_metrics[0].value == 0.0

        gpu0_metrics = [m for m in result.metrics if m.labels.get("gpu") == "0"]
        available = [m for m in gpu0_metrics if m.name == "miles_ft_gpu_available"]
        assert available[0].value == 1.0

    async def test_nvml_init_failure_returns_empty(self) -> None:
        mock = MagicMock()
        mock.nvmlInit.side_effect = RuntimeError("NVML not available")
        with _patched_gpu_collector(mock_pynvml=mock) as (collector, _):
            result = await collector.collect()

        assert result.metrics == []

    async def test_row_remap_pending_value(self) -> None:
        with _patched_gpu_collector(device_count=1, remap_info=(0, 0, 3, 1)) as (collector, _):
            result = await collector.collect()

        pending = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_row_remap_pending"]
        assert len(pending) == 1
        assert pending[0].value == 3.0

        failure = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_row_remap_failure"]
        assert len(failure) == 1
        assert failure[0].value == 1.0

    async def test_close_calls_nvml_shutdown(self) -> None:
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            await collector.close()

        mock.nvmlShutdown.assert_called_once()

    async def test_pcie_bandwidth_conversion(self) -> None:
        with _patched_gpu_collector(device_count=1, pcie_throughput_kb=2097152) as (collector, _):
            result = await collector.collect()

        bw = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_pcie_tx_throughput"]
        assert len(bw) == 1
        assert bw[0].value == pytest.approx(2097152 * 1024)

    async def test_close_safe_when_nvml_unavailable(self) -> None:
        mock = MagicMock()
        mock.nvmlInit.side_effect = RuntimeError("NVML not available")
        with _patched_gpu_collector(mock_pynvml=mock) as (collector, _):
            await collector.close()

        mock.nvmlShutdown.assert_not_called()

    async def test_partial_metric_failure_still_reports_others(self) -> None:
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            mock.nvmlDeviceGetTemperature.side_effect = RuntimeError("temp failed")
            result = await collector.collect()

        names = {m.name for m in result.metrics}
        assert "miles_ft_gpu_available" in names
        assert "miles_ft_dcgm_fi_dev_gpu_temp" not in names
        assert "miles_ft_dcgm_fi_dev_row_remap_pending" in names
        assert "miles_ft_dcgm_fi_dev_pcie_tx_throughput" in names

    async def test_collect_interval_default(self) -> None:
        with _patched_gpu_collector(device_count=0) as (collector, _):
            pass

        assert collector.collect_interval == 10.0

    # P2 item 18: individual collection method edge cases
    async def test_row_remap_failure_returns_empty_for_that_metric(self) -> None:
        """_collect_row_remap raises → graceful_degrade returns [], other metrics still reported."""
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            mock.nvmlDeviceGetRemappedRows.side_effect = RuntimeError("remap query failed")
            result = await collector.collect()

        names = {m.name for m in result.metrics}
        assert "miles_ft_dcgm_fi_dev_row_remap_pending" not in names
        assert "miles_ft_dcgm_fi_dev_row_remap_failure" not in names
        assert "miles_ft_gpu_available" in names
        assert "miles_ft_dcgm_fi_dev_gpu_temp" in names

    async def test_utilization_at_zero_percent(self) -> None:
        """GPU utilization at 0% should be reported correctly."""
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            mock.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=0, memory=0)
            result = await collector.collect()

        util = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_gpu_util"]
        assert len(util) == 1
        assert util[0].value == 0.0

    async def test_utilization_at_100_percent(self) -> None:
        """GPU utilization at 100% should be reported correctly."""
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            mock.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=100, memory=80)
            result = await collector.collect()

        util = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_gpu_util"]
        assert len(util) == 1
        assert util[0].value == 100.0

    async def test_pcie_bandwidth_failure_returns_empty_for_that_metric(self) -> None:
        """_collect_pcie_bandwidth raises → graceful_degrade returns []."""
        with _patched_gpu_collector(device_count=1) as (collector, mock):
            mock.nvmlDeviceGetPcieThroughput.side_effect = RuntimeError("pcie query failed")
            result = await collector.collect()

        names = {m.name for m in result.metrics}
        assert "miles_ft_dcgm_fi_dev_pcie_tx_throughput" not in names
        assert "miles_ft_gpu_available" in names


class TestGpuCollectorRealHardware:
    """Zero-mock tests against real NVML. Run on GPU nodes."""

    @pytest.mark.anyio
    async def test_collect_returns_gpu_metrics(self) -> None:
        collector = GpuCollector()
        result = await collector.collect()
        assert len(result.metrics) > 0

        names = {s.name for s in result.metrics}
        assert "miles_ft_gpu_available" in names
        assert "miles_ft_dcgm_fi_dev_gpu_temp" in names

    @pytest.mark.anyio
    async def test_gpu_temperature_in_sane_range(self) -> None:
        collector = GpuCollector()
        result = await collector.collect()
        for s in result.metrics:
            if s.name == "miles_ft_dcgm_fi_dev_gpu_temp":
                assert 0 < s.value < 120
            if s.name == "miles_ft_gpu_available":
                assert s.value == 1.0

    @pytest.mark.anyio
    async def test_all_metric_values_are_finite(self) -> None:
        collector = GpuCollector()
        result = await collector.collect()
        for s in result.metrics:
            assert math.isfinite(s.value)
