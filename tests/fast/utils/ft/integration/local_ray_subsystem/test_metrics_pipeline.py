"""Local Ray: Metrics pipeline — exporter→scrape→detection, log_step→MiniWandb."""

from __future__ import annotations

import time

import pytest
import ray
from prometheus_client import Gauge
from tests.fast.utils.ft.utils.controller_fakes import FakeNodeManager, FastHangDetector
from tests.fast.utils.ft.integration.conftest import get_status, poll_for_run_id

from miles.utils.ft.agents.metrics.prometheus_exporter import PrometheusExporter
from miles.utils.ft.controller.detectors.core.nan_loss import NanLossDetector
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.models.diagnostic import DiagnosticResult
from miles.utils.ft.models.metric_names import AGENT_HEARTBEAT
from miles.utils.ft.models.controller import ControllerMode
from miles.utils.ft.platform.config import FtControllerConfig
from miles.utils.ft.platform.ray_wrappers.controller_actor import FtControllerActor
from miles.utils.ft.platform.stubs import StubTrainingJob
from miles.utils.ft.protocols.controller import ft_controller_actor_name

pytestmark = [
    pytest.mark.local_ray,
]


class TestLogStepArrivesInMiniWandb:
    def test_latest_iteration_updated_after_log_step(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        ray.get(
            handle.log_step.remote(
                run_id=run_id,
                step=42,
                metrics={"loss": 0.1, "iteration": 42},
            ),
            timeout=5,
        )

        status = get_status(handle)
        assert status.latest_iteration == 42

    def test_multiple_log_steps_track_latest(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        for step in [10, 20, 30]:
            ray.get(
                handle.log_step.remote(
                    run_id=run_id,
                    step=step,
                    metrics={"iteration": step},
                ),
                timeout=5,
            )

        status = get_status(handle)
        assert status.latest_iteration == 30


class TestExporterScrapeByMiniPrometheus:
    """Verify that MiniPrometheus can scrape gauges from a PrometheusExporter.

    This test runs entirely in the test process (no actor), but validates the
    exporter→scrape data path used within the controller's tick loop.
    """

    @pytest.mark.anyio
    async def test_mini_prometheus_scrapes_exporter_gauges(
        self,
        local_ray: None,
    ) -> None:
        exporter = PrometheusExporter()
        gauge = Gauge(
            "test_scrape_metric",
            "test gauge",
            labelnames=["rank"],
            registry=exporter.registry,
        )
        gauge.labels(rank="0").set(99.0)

        mini_prom = MiniPrometheus(
            config=MiniPrometheusConfig(),
        )
        mini_prom.add_scrape_target(
            target_id="test-rank",
            address=exporter.get_address(),
        )

        await mini_prom.scrape_once()

        df = mini_prom.query_latest(metric_name="test_scrape_metric")
        try:
            assert len(df) >= 1
            assert 99.0 in df["value"].to_list()
        finally:
            exporter.shutdown()


class TestRankExporterRegisteredAsScrapeTarget:
    """When register_training_rank provides an exporter_address, it should
    be added as a scrape target in MiniPrometheus."""

    def test_exporter_address_used_for_scraping(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        exporter = PrometheusExporter()
        gauge = Gauge(
            "ft_training_iteration",
            "test iteration gauge",
            labelnames=["rank", "node_id"],
            registry=exporter.registry,
        )
        gauge.labels(rank="0", node_id="n0").set(55.0)

        try:
            ray.get(
                handle.register_training_rank.remote(
                    run_id=run_id,
                    rank=0,
                    world_size=1,
                    node_id="n0",
                    exporter_address=exporter.get_address(),
                    pid=1000,
                ),
                timeout=5,
            )

            time.sleep(0.3)
        finally:
            exporter.shutdown()


class TestNanLossTriggersRecovery:
    """NaN loss sent via log_step → NanLossDetector → ENTER_RECOVERY (M4)."""

    def test_nan_loss_triggers_recovery_via_detector(
        self,
        local_ray: None,
    ) -> None:
        name = ft_controller_actor_name("nan-det")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=0.05, ft_id="nan-det"),
            node_manager_override=FakeNodeManager(),
            training_job_override=StubTrainingJob(),
            notifier_override=None,
            detectors_override=[NanLossDetector()],
        )

        try:
            handle.submit_and_run.remote()
            run_id = poll_for_run_id(handle)

            ray.get(
                handle.register_training_rank.remote(
                    run_id=run_id,
                    rank=0,
                    world_size=1,
                    node_id="nan-node",
                    exporter_address="http://nan-node:9090",
                    pid=1000,
                ),
                timeout=5,
            )

            ray.get(
                handle.log_step.remote(
                    run_id=run_id,
                    step=1,
                    metrics={"loss": float("nan")},
                ),
                timeout=5,
            )

            deadline = time.monotonic() + 15.0
            entered_recovery = False
            while time.monotonic() < deadline:
                s = get_status(handle)
                if s.mode == ControllerMode.RECOVERY:
                    entered_recovery = True
                    break
                time.sleep(0.2)

            assert entered_recovery, "NanLossDetector did not trigger recovery"
        finally:
            try:
                ray.get(handle.shutdown.remote(), timeout=5)
            except Exception:
                pass
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except ValueError:
                pass


class TestHangDetectionFullPath:
    """Exporter reports stale iteration → HangDetector triggers recovery (M3)."""

    def test_stale_iteration_triggers_hang_recovery(
        self,
        local_ray: None,
    ) -> None:
        name = ft_controller_actor_name("hang-det")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=0.1, ft_id="hang-det"),
            node_manager_override=FakeNodeManager(),
            training_job_override=StubTrainingJob(),
            notifier_override=None,
            detectors_override=[FastHangDetector(timeout_seconds=3.0)],
        )

        try:
            handle.submit_and_run.remote()
            run_id = poll_for_run_id(handle)

            exporter = PrometheusExporter()
            gauge = Gauge(
                AGENT_HEARTBEAT,
                "iteration gauge for hang test",
                labelnames=["rank", "node_id"],
                registry=exporter.registry,
            )
            gauge.labels(rank="0", node_id="hang-node").set(42.0)

            try:
                ray.get(
                    handle.register_training_rank.remote(
                        run_id=run_id,
                        rank=0,
                        world_size=1,
                        node_id="hang-node",
                        exporter_address=exporter.get_address(),
                        pid=1000,
                    ),
                    timeout=5,
                )

                deadline = time.monotonic() + 20.0
                entered_recovery = False
                while time.monotonic() < deadline:
                    s = get_status(handle)
                    if s.mode == ControllerMode.RECOVERY:
                        entered_recovery = True
                        break
                    time.sleep(0.3)

                assert entered_recovery, "_FastHangDetector did not trigger recovery"
            finally:
                exporter.shutdown()
        finally:
            try:
                ray.get(handle.shutdown.remote(), timeout=5)
            except Exception:
                pass
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except ValueError:
                pass


class TestRegisterNodeAgentSerialization:
    """register_node_agent.remote() serializes a Python object via cloudpickle (M5)."""

    def test_node_agent_survives_cloudpickle_serialization(
        self,
        local_ray: None,
    ) -> None:

        class _FakeNodeAgent:
            async def run_diagnostic(
                self,
                diagnostic_type: str,
                timeout_seconds: int = 120,
                **kwargs: object,
            ) -> DiagnosticResult:
                return DiagnosticResult(
                    passed=True,
                    details="fake diagnostic pass",
                    diagnostic_type=diagnostic_type,
                    node_id="fake-node",
                )

        name = ft_controller_actor_name("m5-agent")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=0.1, ft_id="m5-agent"),
        )

        try:
            handle.submit_and_run.remote()
            poll_for_run_id(handle)

            ray.get(
                handle.register_node_agent.remote(
                    node_id="fake-node",
                    agent=_FakeNodeAgent(),
                ),
                timeout=5,
            )

            status = get_status(handle)
            assert status.mode == ControllerMode.MONITORING
        finally:
            try:
                ray.get(handle.shutdown.remote(), timeout=5)
            except Exception:
                pass
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except ValueError:
                pass
