from __future__ import annotations

from datetime import timedelta

from prometheus_client import CollectorRegistry

from miles.utils.ft.controller.detectors.base import DetectorContext
from miles.utils.ft.controller.metrics.exporter import ControllerExporter
from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.metrics.mini_wandb import MiniWandb
from miles.utils.ft.models.metric_names import (
    DCGM_FI_DEV_GPU_TEMP,
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    TRAINING_JOB_STATUS,
    XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
)
from miles.utils.ft.models.metrics import CounterSample, GaugeSample
from miles.utils.ft.protocols.platform import JobStatus


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
# Metric inject helpers
# ---------------------------------------------------------------------------


def inject_gpu_unavailable(
    store: MiniPrometheus,
    node_id: str = "node-0",
    gpu: str = "0",
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=GPU_AVAILABLE, labels={"gpu": gpu}, value=0.0),
        ],
    )


def inject_critical_xid(
    store: MiniPrometheus,
    node_id: str = "node-0",
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            CounterSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels={}, delta=1.0),
        ],
    )


def inject_disk_fault(
    store: MiniPrometheus,
    node_id: str = "node-0",
    mountpoint: str = "/data",
    available_bytes: float = 0.0,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_FILESYSTEM_AVAIL_BYTES, labels={"mountpoint": mountpoint}, value=available_bytes),
        ],
    )


def inject_nic_down(
    store: MiniPrometheus,
    node_id: str = "node-0",
    device: str = "ib0",
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"device": device}, value=0.0),
        ],
    )


def inject_nic_up(
    store: MiniPrometheus,
    node_id: str = "node-0",
    device: str = "ib0",
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=NODE_NETWORK_UP, labels={"device": device}, value=1.0),
        ],
    )


def inject_training_job_status(store: MiniPrometheus, status_value: int) -> None:
    store.ingest_samples(
        target_id="controller",
        samples=[
            GaugeSample(name=TRAINING_JOB_STATUS, labels={}, value=float(status_value)),
        ],
    )


def inject_gpu_temperature(
    store: MiniPrometheus,
    node_id: str = "node-0",
    gpu: str = "0",
    celsius: float = 65.0,
) -> None:
    store.ingest_samples(
        target_id=node_id,
        samples=[
            GaugeSample(name=DCGM_FI_DEV_GPU_TEMP, labels={"gpu": gpu}, value=celsius),
        ],
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
