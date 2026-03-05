from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from tests.fast.utils.ft.helpers import make_mock_pynvml

from miles.utils.ft.agents.collectors.gpu import GpuCollector


class TestGpuCollector:
    async def test_normal_8_gpus_produces_48_metrics(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=8)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
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
        mock_pynvml = make_mock_pynvml(device_count=4, failing_handle_indices={2})
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        gpu2_metrics = [m for m in result.metrics if m.labels.get("gpu") == "2"]
        assert len(gpu2_metrics) == 1
        assert gpu2_metrics[0].name == "miles_ft_gpu_available"
        assert gpu2_metrics[0].value == 0.0

        gpu0_metrics = [m for m in result.metrics if m.labels.get("gpu") == "0"]
        available = [m for m in gpu0_metrics if m.name == "miles_ft_gpu_available"]
        assert available[0].value == 1.0

    async def test_nvml_init_failure_returns_empty(self) -> None:
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML not available")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        assert result.metrics == []

    async def test_row_remap_pending_value(self) -> None:
        mock_pynvml = make_mock_pynvml(
            device_count=1,
            remap_info=(0, 0, 3, 1),
        )
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        pending = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_row_remap_pending"]
        assert len(pending) == 1
        assert pending[0].value == 3.0

        failure = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_row_remap_failure"]
        assert len(failure) == 1
        assert failure[0].value == 1.0

    async def test_close_calls_nvml_shutdown(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            await collector.close()

        mock_pynvml.nvmlShutdown.assert_called_once()

    async def test_pcie_bandwidth_conversion(self) -> None:
        mock_pynvml = make_mock_pynvml(
            device_count=1,
            pcie_throughput_kb=2097152,
        )
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        bw = [m for m in result.metrics if m.name == "miles_ft_dcgm_fi_dev_pcie_tx_throughput"]
        assert len(bw) == 1
        assert bw[0].value == pytest.approx(2097152 * 1024)

    async def test_close_safe_when_nvml_unavailable(self) -> None:
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML not available")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            await collector.close()

        mock_pynvml.nvmlShutdown.assert_not_called()

    async def test_partial_metric_failure_still_reports_others(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=1)
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = RuntimeError("temp failed")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        names = {m.name for m in result.metrics}
        assert "miles_ft_gpu_available" in names
        assert "miles_ft_dcgm_fi_dev_gpu_temp" not in names
        assert "miles_ft_dcgm_fi_dev_row_remap_pending" in names
        assert "miles_ft_dcgm_fi_dev_pcie_tx_throughput" in names

    async def test_collect_interval_default(self) -> None:
        mock_pynvml = make_mock_pynvml(device_count=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()

        assert collector.collect_interval == 10.0
