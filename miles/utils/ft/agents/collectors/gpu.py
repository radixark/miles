from __future__ import annotations

import asyncio
import logging

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import CollectorOutput, MetricSample

logger = logging.getLogger(__name__)


class GpuCollector(BaseCollector):
    collect_interval: float = 10.0

    def __init__(self) -> None:
        self._nvml_available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_available = True
        except Exception:
            logger.warning("pynvml unavailable — GpuCollector will report all GPUs as unavailable")

    async def collect(self) -> CollectorOutput:
        metrics = await asyncio.to_thread(self._collect_sync)
        return CollectorOutput(metrics=metrics)

    async def close(self) -> None:
        if self._nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                logger.warning("nvmlShutdown failed", exc_info=True)
            self._nvml_available = False

    def _collect_sync(self) -> list[MetricSample]:
        if not self._nvml_available:
            return self._collect_unavailable()

        import pynvml

        samples: list[MetricSample] = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            logger.warning("nvmlDeviceGetCount failed", exc_info=True)
            return []

        for index in range(device_count):
            gpu_label = {"gpu": str(index)}

            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                logger.warning("Cannot get handle for GPU %d", index)
                samples.append(MetricSample(name="gpu_available", labels=gpu_label, value=0.0))
                continue

            samples.append(MetricSample(name="gpu_available", labels=gpu_label, value=1.0))
            self._collect_temperature(handle, gpu_label, samples)
            self._collect_row_remap(handle, gpu_label, samples)
            self._collect_pcie_bandwidth(handle, gpu_label, samples)
            self._collect_utilization(handle, gpu_label, samples)

        return samples

    def _collect_unavailable(self) -> list[MetricSample]:
        return []

    @staticmethod
    def _collect_temperature(
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            import pynvml

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            samples.append(MetricSample(name="gpu_temperature_celsius", labels=gpu_label, value=float(temp)))
        except Exception:
            logger.warning("Failed to get temperature for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_row_remap(
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            import pynvml

            remap_info = pynvml.nvmlDeviceGetRemappedRows(handle)
            pending = remap_info[2]
            failure = remap_info[3]
            samples.append(MetricSample(name="gpu_row_remap_pending", labels=gpu_label, value=float(pending)))
            samples.append(MetricSample(name="gpu_row_remap_failure", labels=gpu_label, value=float(failure)))
        except Exception:
            logger.warning("Failed to get row remap for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_pcie_bandwidth(
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            import pynvml

            throughput_kb_per_s = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES,
            )
            gbps = throughput_kb_per_s / (1024.0 * 1024.0)
            samples.append(MetricSample(name="gpu_pcie_bandwidth_gbps", labels=gpu_label, value=gbps))
        except Exception:
            logger.warning("Failed to get PCIe bandwidth for GPU %s", gpu_label["gpu"])

    @staticmethod
    def _collect_utilization(
        handle: object,
        gpu_label: dict[str, str],
        samples: list[MetricSample],
    ) -> None:
        try:
            import pynvml

            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            samples.append(MetricSample(name="gpu_tensorcore_utilization", labels=gpu_label, value=float(rates.gpu)))
        except Exception:
            logger.warning("Failed to get utilization for GPU %s", gpu_label["gpu"])
