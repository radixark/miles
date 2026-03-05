from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

from prometheus_client import CollectorRegistry

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.metric_names import (
    DCGM_FI_DEV_GPU_TEMP,
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    TRAINING_JOB_STATUS,
    XID_CODE_RECENT,
)
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.controller_exporter import ControllerExporter
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.diagnostics.base import BaseDiagnostic
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import (
    ActionType,
    CollectorOutput,
    Decision,
    DiagnosticResult,
    MetricSample,
)
from miles.utils.ft.platform.protocols import JobStatus


def get_sample_value(
    registry: CollectorRegistry,
    metric_name: str,
) -> float | None:
    """Read the current value of a metric from a CollectorRegistry."""
    for metric_family in registry.collect():
        for sample in metric_family.samples:
            if sample.name == metric_name:
                return sample.value
    return None


def make_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> MetricSample:
    return MetricSample(name=name, labels=labels or {}, value=value)


def make_fake_metric_store(
    metrics: list[MetricSample] | None = None,
    target_id: str = "node-0",
) -> MiniPrometheus:
    store = MiniPrometheus(config=MiniPrometheusConfig(
        retention=timedelta(minutes=60),
    ))
    if metrics:
        store.ingest_samples(target_id=target_id, samples=metrics)
    return store


def make_fake_mini_wandb(
    steps: dict[int, dict[str, float]] | None = None,
    run_id: str = "test-run",
    rank: int = 0,
) -> MiniWandb:
    wandb = MiniWandb(active_run_id=run_id)
    if steps:
        for step_num, metrics in sorted(steps.items()):
            wandb.log_step(run_id=run_id, rank=rank, step=step_num, metrics=metrics)
    return wandb


EMPTY_RANK_PLACEMENT: dict[int, str] = {}


def make_detector_context(
    metric_store: MiniPrometheus | None = None,
    mini_wandb: MiniWandb | None = None,
    rank_placement: dict[int, str] | None = None,
    job_status: JobStatus = JobStatus.RUNNING,
) -> DetectorContext:
    return DetectorContext(
        metric_store=metric_store or make_fake_metric_store(),
        mini_wandb=mini_wandb or make_fake_mini_wandb(),
        rank_placement=rank_placement if rank_placement is not None else {},
        job_status=job_status,
    )


# ---------------------------------------------------------------------------
# Detector test helpers — inject functions (detectors milestone)
# ---------------------------------------------------------------------------


def inject_gpu_unavailable(
    store: MiniPrometheus, node_id: str = "node-0", gpu: str = "0",
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=GPU_AVAILABLE, labels={"gpu": gpu}, value=0.0),
    ])


def inject_critical_xid(
    store: MiniPrometheus, node_id: str = "node-0", xid_code: int = 48,
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=XID_CODE_RECENT, labels={"xid": str(xid_code)}, value=1.0),
    ])


def inject_disk_fault(
    store: MiniPrometheus,
    node_id: str = "node-0",
    mountpoint: str = "/data",
    available_bytes: float = 0.0,
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": mountpoint}, value=available_bytes),
    ])


def inject_nic_down(
    store: MiniPrometheus, node_id: str = "node-0", device: str = "ib0",
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=NODE_NETWORK_UP, labels={"device": device}, value=0.0),
    ])


def inject_nic_up(
    store: MiniPrometheus, node_id: str = "node-0", device: str = "ib0",
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=NODE_NETWORK_UP, labels={"device": device}, value=1.0),
    ])


def inject_training_job_status(store: MiniPrometheus, status_value: int) -> None:
    store.ingest_samples(target_id="controller", samples=[
        MetricSample(name=TRAINING_JOB_STATUS, labels={}, value=float(status_value)),
    ])


def inject_gpu_temperature(
    store: MiniPrometheus,
    node_id: str = "node-0",
    gpu: str = "0",
    celsius: float = 65.0,
) -> None:
    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=DCGM_FI_DEV_GPU_TEMP, labels={"gpu": gpu}, value=celsius),
    ])


def inject_healthy_node(
    store: MiniPrometheus,
    node_id: str = "node-0",
    num_gpus: int = 8,
    num_nics: int = 4,
) -> None:
    for i in range(num_gpus):
        store.ingest_samples(target_id=node_id, samples=[
            MetricSample(name=GPU_AVAILABLE, labels={"gpu": str(i)}, value=1.0),
            MetricSample(name=DCGM_FI_DEV_GPU_TEMP, labels={"gpu": str(i)}, value=65.0),
        ])

    for i in range(num_nics):
        store.ingest_samples(target_id=node_id, samples=[
            MetricSample(name=NODE_NETWORK_UP, labels={"device": f"ib{i}"}, value=1.0),
        ])

    store.ingest_samples(target_id=node_id, samples=[
        MetricSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/data"}, value=500e9),
    ])


# ---------------------------------------------------------------------------
# Platform fakes (controller-skeleton milestone)
# ---------------------------------------------------------------------------


class FakeNodeManager:
    """In-memory implementation of NodeManagerProtocol for testing."""

    def __init__(self) -> None:
        self._bad_nodes: set[str] = set()

    async def mark_node_bad(self, node_id: str, reason: str = "") -> None:
        self._bad_nodes.add(node_id)

    async def unmark_node_bad(self, node_id: str) -> None:
        self._bad_nodes.discard(node_id)

    def is_node_bad(self, node_id: str) -> bool:
        return node_id in self._bad_nodes

    async def get_bad_nodes(self) -> list[str]:
        return sorted(self._bad_nodes)


class FakeNotifier:
    """Records all send() calls for assertion in tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def send(self, title: str, content: str, severity: str) -> None:
        self.calls.append((title, content, severity))


class FakeTrainingJob:
    """Programmable implementation of TrainingJobProtocol for testing."""

    def __init__(self, status_sequence: list[JobStatus] | None = None) -> None:
        self._status_sequence = status_sequence or [JobStatus.RUNNING]
        self._call_count: int = 0
        self._stopped: bool = False
        self._submitted: bool = False
        self._run_id: str = "fake-initial"

    async def get_training_status(self) -> JobStatus:
        index = min(self._call_count, len(self._status_sequence) - 1)
        status = self._status_sequence[index]
        self._call_count += 1
        return status

    async def stop_training(self, timeout_seconds: int = 300) -> None:
        self._stopped = True

    async def submit_training(self) -> str:
        self._submitted = True
        self._call_count = 0
        self._run_id = f"fake-{id(self)}"
        return self._run_id


# ---------------------------------------------------------------------------
# Test detectors (controller-skeleton milestone)
# ---------------------------------------------------------------------------


class FixedDecisionDetector(BaseFaultDetector):
    """Detector that always returns a fixed Decision. Tracks call count."""

    def __init__(self, decision: Decision) -> None:
        self.call_count = 0
        self._decision = decision

    def evaluate(self, ctx: DetectorContext) -> Decision:
        self.call_count += 1
        return self._decision


_ALWAYS_NONE_DECISION = Decision(action=ActionType.NONE, reason="always none")
_ALWAYS_MARK_BAD_DECISION = Decision(
    action=ActionType.MARK_BAD_AND_RESTART,
    bad_node_ids=["node-1"],
    reason="test fault detected",
)


def AlwaysNoneDetector() -> FixedDecisionDetector:
    return FixedDecisionDetector(decision=_ALWAYS_NONE_DECISION)


def AlwaysMarkBadDetector() -> FixedDecisionDetector:
    return FixedDecisionDetector(decision=_ALWAYS_MARK_BAD_DECISION)


# ---------------------------------------------------------------------------
# Controller factory (controller-skeleton milestone)
# ---------------------------------------------------------------------------


class ControllerTestHarness(NamedTuple):
    controller: FtController
    node_manager: FakeNodeManager
    training_job: FakeTrainingJob
    metric_store: MiniPrometheus
    mini_wandb: MiniWandb
    controller_exporter: ControllerExporter
    notifier: FakeNotifier | None


def make_test_controller(
    detectors: list[BaseFaultDetector] | None = None,
    status_sequence: list[JobStatus] | None = None,
    notifier: FakeNotifier | None = FakeNotifier,
    tick_interval: float = 0.01,
    controller_exporter: ControllerExporter | None = None,
    diagnostic_scheduler: object | None = None,
) -> ControllerTestHarness:
    """Construct a Controller and all its dependencies for testing.

    ``notifier`` defaults to a fresh FakeNotifier instance. Pass ``None``
    explicitly to create a Controller without a notifier.

    ``diagnostic_scheduler`` defaults to a real DiagnosticScheduler with
    empty pipeline (same behavior as old stub). Pass a FakeDiagnosticScheduler
    for recovery-specific tests.
    """
    real_notifier: FakeNotifier | None = FakeNotifier() if notifier is FakeNotifier else notifier
    node_manager = FakeNodeManager()
    training_job = FakeTrainingJob(status_sequence=status_sequence)
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()
    if controller_exporter is None:
        controller_exporter = ControllerExporter(registry=CollectorRegistry())
    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        notifier=real_notifier,
        detectors=detectors,
        tick_interval=tick_interval,
        controller_exporter=controller_exporter,
        scrape_target_manager=metric_store,
        diagnostic_scheduler=diagnostic_scheduler,
    )
    return ControllerTestHarness(
        controller=controller,
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        controller_exporter=controller_exporter,
        notifier=real_notifier,
    )


# ---------------------------------------------------------------------------
# Diagnostic test helpers (diag-framework milestone)
# ---------------------------------------------------------------------------


class StubDiagnostic(BaseDiagnostic):
    """Test diagnostic that returns a configurable pass/fail result."""

    diagnostic_type = "stub"

    def __init__(
        self, passed: bool = True, details: str = "stub passed",
    ) -> None:
        self._passed = passed
        self._details = details

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=self._passed,
            details=self._details,
        )


class FailingDiagnostic(BaseDiagnostic):
    """Test diagnostic that always reports failure."""

    diagnostic_type = "failing"

    def __init__(self, details: str = "diagnostic failed") -> None:
        self._details = details

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=False,
            details=self._details,
        )


class SlowDiagnostic(BaseDiagnostic):
    """Test diagnostic that sleeps longer than its timeout."""

    diagnostic_type = "slow"

    def __init__(self, sleep_seconds: float = 300.0) -> None:
        self._sleep_seconds = sleep_seconds

    async def run(
        self, node_id: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        await asyncio.sleep(self._sleep_seconds)
        return DiagnosticResult(
            diagnostic_type=self.diagnostic_type,
            node_id=node_id,
            passed=True,
            details="should not reach here",
        )


# ---------------------------------------------------------------------------
# Diagnostic scheduler fakes (recovery milestone)
# ---------------------------------------------------------------------------


class FakeDiagnosticScheduler:
    """Programmable stub for DiagnosticScheduler in recovery tests."""

    def __init__(self, decision: Decision | None = None) -> None:
        self._decision = decision or Decision(
            action=ActionType.NOTIFY_HUMAN,
            reason="fake diagnostic — all passed",
        )
        self.call_count: int = 0
        self.last_trigger_reason: str | None = None
        self.last_suspect_node_ids: list[str] | None = None

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: str,
        suspect_node_ids: list[str] | None = None,
    ) -> Decision:
        self.call_count += 1
        self.last_trigger_reason = trigger_reason
        self.last_suspect_node_ids = suspect_node_ids
        return self._decision


# ---------------------------------------------------------------------------
# Agent test helpers (agent-skeleton milestone)
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


def make_fake_agents(
    node_results: dict[str, dict[str, bool]],
) -> dict[str, "FakeNodeAgent"]:
    """Build FakeNodeAgents from {node_id: {diag_type: passed}} mapping."""
    agents: dict[str, FakeNodeAgent] = {}
    for node_id, results in node_results.items():
        diagnostic_results = {
            diag_type: DiagnosticResult(
                diagnostic_type=diag_type,
                node_id=node_id,
                passed=passed,
                details="pass" if passed else "fail",
            )
            for diag_type, passed in results.items()
        }
        agents[node_id] = FakeNodeAgent(diagnostic_results=diagnostic_results)
    return agents


class FakeNodeAgent:
    def __init__(
        self,
        diagnostic_results: dict[str, DiagnosticResult] | None = None,
    ) -> None:
        self._diagnostic_results = diagnostic_results or {}
        self.cleanup_called: bool = False
        self.cleanup_job_id: str | None = None

    async def run_diagnostic(
        self, diagnostic_type: str, timeout_seconds: int = 120,
    ) -> DiagnosticResult:
        result = self._diagnostic_results.get(diagnostic_type)
        if result is None:
            return DiagnosticResult(
                diagnostic_type=diagnostic_type,
                node_id="fake",
                passed=False,
                details=f"unknown diagnostic type: {diagnostic_type}",
            )
        return result

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        self.cleanup_called = True
        self.cleanup_job_id = training_job_id


# ---------------------------------------------------------------------------
# hw-collectors test helpers
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
