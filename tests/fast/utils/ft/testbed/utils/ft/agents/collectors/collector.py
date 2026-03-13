from __future__ import annotations

import logging

import ray
from tests.fast.utils.ft.utils.training_simulator import CollectorStateActor

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.types import GaugeSample, MetricSample
from miles.utils.ft.utils.metric_names import (
    DCGM_FI_DEV_GPU_TEMP,
    DCGM_FI_DEV_GPU_UTIL,
    GPU_AVAILABLE,
    NODE_FILESYSTEM_AVAIL_BYTES,
    NODE_NETWORK_UP,
    XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL,
)

logger = logging.getLogger(__name__)


class TestbedCollector(BaseCollector):
    """Collector that produces realistic baseline metrics with anomaly injection.

    Baseline metrics simulate a healthy GPU node. Tests inject anomalies
    (e.g. XID errors) via a CollectorStateActor that overrides baseline values.
    """

    collect_interval = 0.3

    def __init__(
        self,
        node_id: str,
        state_actor: ray.actor.ActorHandle,
    ) -> None:
        self._node_id = node_id
        self._state_actor = state_actor
        self._counter = 0

    @classmethod
    def create(cls, node_id: str) -> tuple[TestbedCollector, ray.actor.ActorHandle]:
        state_actor = CollectorStateActor.remote()
        collector = cls(node_id=node_id, state_actor=state_actor)
        return collector, state_actor

    def _collect_sync(self) -> list[MetricSample]:
        self._counter += 1
        baseline = self._build_baseline_metrics()

        injected: list[MetricSample] = []
        try:
            injected = ray.get(self._state_actor.get_metrics.remote(), timeout=1.0)
        except Exception:
            logger.debug("Failed to read injected metrics", exc_info=True)

        if injected:
            injected_names = {m.name for m in injected}
            baseline = [m for m in baseline if m.name not in injected_names]
            baseline.extend(injected)

        return baseline

    def _build_baseline_metrics(self) -> list[GaugeSample]:
        labels = {"node_id": self._node_id}
        return [
            GaugeSample(name=GPU_AVAILABLE, labels=labels, value=1.0),
            GaugeSample(name=DCGM_FI_DEV_GPU_TEMP, labels=labels, value=45.0),
            GaugeSample(name=DCGM_FI_DEV_GPU_UTIL, labels=labels, value=0.0),
            GaugeSample(name=XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL, labels=labels, value=0.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={**labels, "device": "eth0"}, value=1.0),
            GaugeSample(name=NODE_NETWORK_UP, labels={**labels, "device": "eth1"}, value=1.0),
            GaugeSample(
                name=NODE_FILESYSTEM_AVAIL_BYTES, labels={**labels, "mountpoint": "/"}, value=500_000_000_000.0
            ),
        ]
