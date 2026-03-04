from datetime import timedelta
from typing import NamedTuple

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.detectors.base import BaseFaultDetector
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

    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
    ) -> Decision:
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


def make_test_controller(
    detectors: list[BaseFaultDetector] | None = None,
    status_sequence: list[JobStatus] | None = None,
    tick_interval: float = 0.01,
) -> ControllerTestHarness:
    """Construct a Controller and all its dependencies for testing."""
    node_manager = FakeNodeManager()
    training_job = FakeTrainingJob(status_sequence=status_sequence)
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()
    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        detectors=detectors,
        tick_interval=tick_interval,
    )
    return ControllerTestHarness(
        controller=controller,
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
    )


# ---------------------------------------------------------------------------
# Agent test helpers (agent-skeleton milestone)
# ---------------------------------------------------------------------------


class TestCollector(BaseCollector):
    def __init__(self, metrics: list[MetricSample] | None = None) -> None:
        self._metrics = metrics or []

    def set_metrics(self, metrics: list[MetricSample]) -> None:
        self._metrics = metrics

    async def collect(self) -> CollectorOutput:
        return CollectorOutput(metrics=self._metrics)


class FakeNodeAgent:
    def __init__(
        self,
        diagnostic_results: dict[str, DiagnosticResult] | None = None,
    ) -> None:
        self._diagnostic_results = diagnostic_results or {}
        self.cleanup_called: bool = False
        self.cleanup_job_id: str | None = None

    async def run_diagnostic(self, diagnostic_type: str) -> DiagnosticResult:
        return self._diagnostic_results[diagnostic_type]

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        self.cleanup_called = True
        self.cleanup_job_id = training_job_id
