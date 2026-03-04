from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.collectors.gpu import GpuCollector


def _make_mock_pynvml(
    device_count: int = 8,
    temperature: int = 65,
    remap_info: tuple[int, int, int, int] = (0, 0, 0, 0),
    pcie_throughput_kb: int = 1048576,
    utilization_gpu: int = 50,
    failing_handle_indices: set[int] | None = None,
) -> MagicMock:
    failing = failing_handle_indices or set()
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    mock.NVML_PCIE_UTIL_TX_BYTES = 1

    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None
    mock.nvmlDeviceGetCount.return_value = device_count

    def get_handle(index: int) -> object:
        if index in failing:
            raise RuntimeError(f"GPU {index} handle failed")
        return f"handle-{index}"

    mock.nvmlDeviceGetHandleByIndex.side_effect = get_handle
    mock.nvmlDeviceGetTemperature.return_value = temperature
    mock.nvmlDeviceGetRemappedRows.return_value = remap_info
    mock.nvmlDeviceGetPcieThroughput.return_value = pcie_throughput_kb
    mock.nvmlDeviceGetUtilizationRates.return_value = SimpleNamespace(gpu=utilization_gpu)

    return mock


class TestGpuCollector:
    @pytest.mark.asyncio()
    async def test_normal_8_gpus_produces_48_metrics(self) -> None:
        mock_pynvml = _make_mock_pynvml(device_count=8)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        assert len(result.metrics) == 48  # 8 GPUs × 6 metrics

        gpu0_names = {m.name for m in result.metrics if m.labels.get("gpu") == "0"}
        assert gpu0_names == {
            "gpu_available",
            "gpu_temperature_celsius",
            "gpu_row_remap_pending",
            "gpu_row_remap_failure",
            "gpu_pcie_bandwidth_gbps",
            "gpu_tensorcore_utilization",
        }

    @pytest.mark.asyncio()
    async def test_failing_handle_reports_gpu_unavailable(self) -> None:
        mock_pynvml = _make_mock_pynvml(device_count=4, failing_handle_indices={2})
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        gpu2_metrics = [m for m in result.metrics if m.labels.get("gpu") == "2"]
        assert len(gpu2_metrics) == 1
        assert gpu2_metrics[0].name == "gpu_available"
        assert gpu2_metrics[0].value == 0.0

        gpu0_metrics = [m for m in result.metrics if m.labels.get("gpu") == "0"]
        available = [m for m in gpu0_metrics if m.name == "gpu_available"]
        assert available[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_nvml_init_failure_returns_empty(self) -> None:
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML not available")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        assert result.metrics == []

    @pytest.mark.asyncio()
    async def test_row_remap_pending_value(self) -> None:
        mock_pynvml = _make_mock_pynvml(
            device_count=1,
            remap_info=(0, 0, 3, 1),
        )
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        pending = [m for m in result.metrics if m.name == "gpu_row_remap_pending"]
        assert len(pending) == 1
        assert pending[0].value == 3.0

        failure = [m for m in result.metrics if m.name == "gpu_row_remap_failure"]
        assert len(failure) == 1
        assert failure[0].value == 1.0

    @pytest.mark.asyncio()
    async def test_close_calls_nvml_shutdown(self) -> None:
        mock_pynvml = _make_mock_pynvml(device_count=1)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            await collector.close()

        mock_pynvml.nvmlShutdown.assert_called_once()

    @pytest.mark.asyncio()
    async def test_pcie_bandwidth_conversion(self) -> None:
        mock_pynvml = _make_mock_pynvml(
            device_count=1,
            pcie_throughput_kb=2097152,
        )
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()
            result = await collector.collect()

        bw = [m for m in result.metrics if m.name == "gpu_pcie_bandwidth_gbps"]
        assert len(bw) == 1
        assert bw[0].value == pytest.approx(2.0)

    @pytest.mark.asyncio()
    async def test_collect_interval_default(self) -> None:
        mock_pynvml = _make_mock_pynvml(device_count=0)
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            collector = GpuCollector()

        assert collector.collect_interval == 10.0
