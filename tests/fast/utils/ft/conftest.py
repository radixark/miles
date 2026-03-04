from datetime import datetime, timedelta

from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import MetricSample
from miles.utils.ft.platform.protocols import JobStatus


# ---------------------------------------------------------------------------
# Metric helpers (from mini-prom milestone)
# ---------------------------------------------------------------------------


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
