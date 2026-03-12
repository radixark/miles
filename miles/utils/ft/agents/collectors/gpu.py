from __future__ import annotations

import logging
from types import ModuleType

import miles.utils.ft.utils.metric_names as mn
from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class GpuCollector(BaseCollector):
    def __init__(self) -> None:
        self._pynvml: ModuleType | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except Exception:
            logger.warning("pynvml unavailable — GpuCollector will report all GPUs as unavailable", exc_info=True)

    async def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                logger.warning("nvmlShutdown failed", exc_info=True)
            self._pynvml = None

    def _collect_sync(self) -> list[GaugeSample]:
        pynvml = self._pynvml
        if pynvml is None:
            return []

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            logger.warning("nvmlDeviceGetCount failed", exc_info=True)
            return []

        samples: list[GaugeSample] = []
        for index in range(device_count):
            gpu_label = {"gpu": str(index)}

            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                logger.warning("Cannot get handle for GPU %d", index, exc_info=True)
                samples.append(GaugeSample(name=mn.GPU_AVAILABLE, labels=gpu_label, value=0.0))
                continue

            samples.append(GaugeSample(name=mn.GPU_AVAILABLE, labels=gpu_label, value=1.0))
            samples.extend(self._collect_temperature(pynvml, handle, gpu_label))
            samples.extend(self._collect_row_remap(pynvml, handle, gpu_label))
            samples.extend(self._collect_pcie_bandwidth(pynvml, handle, gpu_label))
            samples.extend(self._collect_utilization(pynvml, handle, gpu_label))

        return samples

    @graceful_degrade(default=[])
    def _collect_temperature(
        self,
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
    ) -> list[GaugeSample]:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return [GaugeSample(name=mn.DCGM_FI_DEV_GPU_TEMP, labels=gpu_label, value=float(temp))]

    @graceful_degrade(default=[])
    def _collect_row_remap(
        self,
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
    ) -> list[GaugeSample]:
        _correctable, _uncorrectable, pending, failure = pynvml.nvmlDeviceGetRemappedRows(handle)
        return [
            GaugeSample(name=mn.DCGM_FI_DEV_ROW_REMAP_PENDING, labels=gpu_label, value=float(pending)),
            GaugeSample(name=mn.DCGM_FI_DEV_ROW_REMAP_FAILURE, labels=gpu_label, value=float(failure)),
        ]

    @graceful_degrade(default=[])
    def _collect_pcie_bandwidth(
        self,
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
    ) -> list[GaugeSample]:
        throughput_kb_per_s = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        bytes_per_s = throughput_kb_per_s * 1024
        return [GaugeSample(name=mn.DCGM_FI_DEV_PCIE_TX_THROUGHPUT, labels=gpu_label, value=float(bytes_per_s))]

    @graceful_degrade(default=[])
    def _collect_utilization(
        self,
        pynvml: ModuleType,
        handle: object,
        gpu_label: dict[str, str],
    ) -> list[GaugeSample]:
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return [GaugeSample(name=mn.DCGM_FI_DEV_GPU_UTIL, labels=gpu_label, value=float(rates.gpu))]
