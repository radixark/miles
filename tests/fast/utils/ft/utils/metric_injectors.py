from __future__ import annotations

from datetime import datetime, timedelta

from prometheus_client import CollectorRegistry

from miles.utils.ft.adapters.types import JobStatus
from miles.utils.ft.agents.types import CounterSample, GaugeSample
from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.utils.metric_names import (
    AGENT_HEARTBEAT,
    DCGM_FI_DEV_GPU_TEMP,
    GPU_AVAILABLE,
    MAIN_JOB_STATUS,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    ROLLOUT_CELL_ALIVE,
    TRAINING_PHASE,
    XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
)
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.controller.types import MetricStore


def get_sample_value(
    registry: CollectorRegistry,
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    """Read the current value of a metric from a CollectorRegistry."""
    for metric_family in registry.collect():
        for sample in metric_family.samples:
            if sample.name != metric_name:
                continue
            if labels is not None and dict(sample.labels) != labels:
                continue
            return sample.value
    return None


def make_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> GaugeSample:
    return GaugeSample(name=name, labels=labels or {}, value=value)


def make_fake_metric_store(
    metrics: list[GaugeSample] | None = None,
    target_id: str = "node-0",
) -> MiniPrometheus:
    store = MiniPrometheus(
        config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        )
    )
    if metrics:
        store.ingest_samples(target_id=target_id, samples=metrics)
    return store


def make_fake_mini_wandb(
    steps: dict[int, dict[str, float]] | None = None,
    run_id: str = "test-run",
) -> MiniWandb:
    wandb = MiniWandb(active_run_id=run_id)
    if steps:
        for step_num, metrics in sorted(steps.items()):
            wandb.log_step(run_id=run_id, step=step_num, metrics=metrics)
    return wandb


def make_test_exporter() -> tuple[CollectorRegistry, ControllerExporter]:
    registry = CollectorRegistry()
    exporter = ControllerExporter(registry=registry)
    return registry, exporter


def make_detector_context(
    metric_store: MiniPrometheus | None = None,
    mini_wandb: MiniWandb | None = None,
    active_node_ids: frozenset[str] | set[str] | None = None,
    job_status: JobStatus = JobStatus.RUNNING,
    active_run_id: str | None = None,
) -> DetectorContext:
    return DetectorContext(
        metric_store=MetricStore(
            time_series_store=metric_store or make_fake_metric_store(),
            mini_wandb=mini_wandb or make_fake_mini_wandb(),
        ),
        active_node_ids=frozenset(active_node_ids) if active_node_ids is not None else frozenset({"node-0"}),
        job_status=job_status,
        active_run_id=active_run_id,
    )


# ---------------------------------------------------------------------------
# Metric inject helpers
# ---------------------------------------------------------------------------


def inject_gpu_unavailable(
    store: MiniPrometheus,
    node_id: str = "node-0",
    gpu: str = "0",
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=GPU_AVAILABLE, labels={"gpu": gpu}, value=0.0),
        ],
        timestamp=timestamp,
    )


def inject_critical_xid(
    store: MiniPrometheus,
    node_id: str = "node-0",
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
        ],
        timestamp=timestamp,
    )


def inject_disk_fault(
    store: MiniPrometheus,
    node_id: str = "node-0",
    mountpoint: str = "/data",
    available_bytes: float = 0.0,
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": mountpoint}, value=available_bytes),
        ],
        timestamp=timestamp,
    )


def inject_nic_down(
    store: MiniPrometheus,
    node_id: str = "node-0",
    device: str = "ib0",
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"device": device}, value=0.0),
        ],
        timestamp=timestamp,
    )


def inject_nic_up(
    store: MiniPrometheus,
    node_id: str = "node-0",
    device: str = "ib0",
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"device": device}, value=1.0),
        ],
        timestamp=timestamp,
    )


def inject_main_job_status(
    store: MiniPrometheus,
    status_value: int,
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id="controller",
        samples=[
            GaugeSample(name=MAIN_JOB_STATUS, labels={}, value=float(status_value)),
        ],
        timestamp=timestamp,
    )


def inject_gpu_temperature(
    store: MiniPrometheus,
    node_id: str = "node-0",
    gpu: str = "0",
    celsius: float = 65.0,
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=DCGM_FI_DEV_GPU_TEMP, labels={"gpu": gpu}, value=celsius),
        ],
        timestamp=timestamp,
    )


def inject_heartbeat(
    store: MiniPrometheus,
    value: float,
    rank: str = "0",
    timestamp: datetime | None = None,
    ft_run_id: str = "",
) -> None:
    labels: dict[str, str] = {"rank": rank}
    if ft_run_id:
        labels["ft_run_id"] = ft_run_id
    store.ingest_samples(
        target_id=f"rank-{rank}",
        samples=[GaugeSample(name=AGENT_HEARTBEAT, labels=labels, value=value)],
        timestamp=timestamp,
    )


def inject_training_phase(
    store: MiniPrometheus,
    phase: float,
    rank: str = "0",
    timestamp: datetime | None = None,
    ft_run_id: str = "",
) -> None:
    labels: dict[str, str] = {"rank": rank}
    if ft_run_id:
        labels["ft_run_id"] = ft_run_id
    store.ingest_samples(
        target_id=f"rank-{rank}",
        samples=[GaugeSample(name=TRAINING_PHASE, labels=labels, value=phase)],
        timestamp=timestamp,
    )


def inject_rollout_cell_alive(
    store: MiniPrometheus,
    cell_id: str,
    alive: bool,
    timestamp: datetime | None = None,
) -> None:
    store.ingest_samples(
        target_id="rollout-ft-agent",
        samples=[GaugeSample(name=ROLLOUT_CELL_ALIVE, value=1.0 if alive else 0.0, labels={"cell_id": cell_id})],
        timestamp=timestamp,
    )


def inject_healthy_node(
    store: MiniPrometheus,
    node_id: str = "node-0",
    num_gpus: int = 8,
    num_nics: int = 4,
) -> None:
    for i in range(num_gpus):
        store.ingest_samples(
            target_id=node_id,
            samples=[
                GaugeSample(name=GPU_AVAILABLE, labels={"gpu": str(i)}, value=1.0),
                GaugeSample(name=DCGM_FI_DEV_GPU_TEMP, labels={"gpu": str(i)}, value=65.0),
            ],
        )

    for i in range(num_nics):
        store.ingest_samples(
            target_id=node_id,
            samples=[
                GaugeSample(name=NODE_NETWORK_UP, labels={"device": f"ib{i}"}, value=1.0),
            ],
        )

    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": "/data"}, value=500e9),
        ],
    )
