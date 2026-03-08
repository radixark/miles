from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.diagnostics.stack_trace import PySpyFrame, PySpyThread
from miles.utils.ft.models.diagnostics import DiagnosticResult
from miles.utils.ft.models.metrics import MetricSample


# ---------------------------------------------------------------------------
# Collector test helpers
# ---------------------------------------------------------------------------


class TestCollector(BaseCollector):
    def __init__(
        self,
        metrics: list[MetricSample] | None = None,
        collect_interval: float = 10.0,
    ) -> None:
        self._metrics = metrics or []
        self.collect_interval = collect_interval

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    def _collect_sync(self) -> list[MetricSample]:
        return list(self._metrics)


class FailingCollector(BaseCollector):
    """Collector that always raises on collect. Tracks call count."""

    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.call_count = 0

    def _collect_sync(self) -> list[MetricSample]:
        self.call_count += 1
        raise RuntimeError("collect failed")


class FailingCloseCollector(BaseCollector):
    """Collector whose close() always raises. Tracks whether close was called."""

    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.close_called = False

    def _collect_sync(self) -> list[MetricSample]:
        return []

    async def close(self) -> None:
        self.close_called = True
        raise RuntimeError("close failed")


# ---------------------------------------------------------------------------
# HW-collector test helpers
# ---------------------------------------------------------------------------


class FakeKmsgReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._consumed = False

    def read_new_lines(self) -> list[str]:
        if self._consumed:
            return []
        self._consumed = True
        return list(self._lines)

    def close(self) -> None:
        pass


def make_mock_pynvml(
    device_count: int = 8,
    temperature: int = 65,
    remap_info: tuple[int, int, int, int] = (0, 0, 0, 0),
    pcie_throughput_kb: int = 1048576,
    utilization_gpu: int = 50,
    failing_handle_indices: set[int] | None = None,
    ecc_uncorrectable: int = 0,
    retired_pages: list[object] | None = None,
    power_state: int = 0,
) -> MagicMock:
    failing = failing_handle_indices or set()
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    mock.NVML_PCIE_UTIL_TX_BYTES = 1
    mock.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    mock.NVML_VOLATILE_ECC_COUNTER_TYPE = 0
    mock.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 0

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
    mock.nvmlDeviceGetTotalEccErrors.return_value = ecc_uncorrectable
    mock.nvmlDeviceGetRetiredPages.return_value = retired_pages or []
    mock.nvmlDeviceGetPowerState.return_value = power_state

    return mock


def create_sysfs_interface(
    base: Path,
    name: str,
    operstate: str = "up",
    rx_errors: int = 0,
    tx_errors: int = 0,
    rx_dropped: int = 0,
    tx_dropped: int = 0,
) -> None:
    iface_dir = base / name
    iface_dir.mkdir(parents=True, exist_ok=True)

    (iface_dir / "operstate").write_text(operstate + "\n")

    stats_dir = iface_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    (stats_dir / "rx_errors").write_text(str(rx_errors) + "\n")
    (stats_dir / "tx_errors").write_text(str(tx_errors) + "\n")
    (stats_dir / "rx_dropped").write_text(str(rx_dropped) + "\n")
    (stats_dir / "tx_dropped").write_text(str(tx_dropped) + "\n")


# ---------------------------------------------------------------------------
# Stack trace test helpers
# ---------------------------------------------------------------------------

_WORKER_THREAD = PySpyThread(
    thread_name="WorkerThread-1",
    active=False,
    owns_gil=False,
    frames=[
        PySpyFrame(name="wait", filename="threading.py", line=320),
        PySpyFrame(name="get", filename="queue.py", line=171),
        PySpyFrame(name="_worker", filename="concurrent/futures/thread.py", line=83),
    ],
)

SAMPLE_PYSPY_THREADS_NORMAL: list[PySpyThread] = [
    PySpyThread(
        thread_name="MainThread",
        active=True,
        owns_gil=False,
        frames=[
            PySpyFrame(name="_wait_for_data", filename="selectors.py", line=451),
            PySpyFrame(name="select", filename="selectors.py", line=469),
            PySpyFrame(name="_run_once", filename="asyncio/base_events.py", line=1922),
            PySpyFrame(name="run_forever", filename="asyncio/base_events.py", line=604),
        ],
    ),
    _WORKER_THREAD,
]

SAMPLE_PYSPY_THREADS_STUCK: list[PySpyThread] = [
    PySpyThread(
        thread_name="MainThread",
        active=True,
        owns_gil=False,
        frames=[
            PySpyFrame(name="nccl_allreduce", filename="nccl_ops.py", line=42),
            PySpyFrame(name="all_reduce", filename="torch/distributed/distributed_c10d.py", line=1234),
            PySpyFrame(name="forward", filename="model.py", line=100),
            PySpyFrame(name="train_step", filename="train.py", line=50),
        ],
    ),
    _WORKER_THREAD,
]

SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK: list[PySpyThread] = [
    PySpyThread(
        thread_name="MainThread",
        active=True,
        owns_gil=False,
        frames=[
            PySpyFrame(name="recv", filename="socket.py", line=123),
            PySpyFrame(name="_receive_data", filename="network.py", line=456),
            PySpyFrame(name="fetch_batch", filename="dataloader.py", line=78),
        ],
    ),
    _WORKER_THREAD,
]

SAMPLE_PYSPY_THREADS_STUCK_FROM_BACKWARD: list[PySpyThread] = [
    PySpyThread(
        thread_name="MainThread",
        active=True,
        owns_gil=False,
        frames=[
            PySpyFrame(name="nccl_allreduce", filename="nccl_ops.py", line=42),
            PySpyFrame(name="all_reduce", filename="torch/distributed/distributed_c10d.py", line=1234),
            PySpyFrame(name="backward", filename="model.py", line=200),
            PySpyFrame(name="train_step", filename="train.py", line=55),
        ],
    ),
    _WORKER_THREAD,
]


def serialize_pyspy_threads(threads: list[PySpyThread]) -> str:
    return json.dumps([t.model_dump() for t in threads])


SAMPLE_PYSPY_JSON_NORMAL = serialize_pyspy_threads(SAMPLE_PYSPY_THREADS_NORMAL)
SAMPLE_PYSPY_JSON_STUCK = serialize_pyspy_threads(SAMPLE_PYSPY_THREADS_STUCK)
SAMPLE_PYSPY_JSON_DIFFERENT_STUCK = serialize_pyspy_threads(SAMPLE_PYSPY_THREADS_DIFFERENT_STUCK)
SAMPLE_PYSPY_JSON_STUCK_FROM_BACKWARD = serialize_pyspy_threads(SAMPLE_PYSPY_THREADS_STUCK_FROM_BACKWARD)


def make_rank_pids_provider(
    mapping: dict[str, dict[int, int]],
) -> Callable[[str], dict[int, int]]:
    def provider(node_id: str) -> dict[int, int]:
        return mapping.get(node_id, {})

    return provider


def make_trace_result(
    node_id: str,
    passed: bool = True,
    details: str = "[]",
) -> DiagnosticResult:
    return DiagnosticResult(
        diagnostic_type="stack_trace",
        node_id=node_id,
        passed=passed,
        details=details,
    )
