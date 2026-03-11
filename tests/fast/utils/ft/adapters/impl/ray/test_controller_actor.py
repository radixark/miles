from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from tests.fast.utils.ft.conftest import make_test_controller

from miles.utils.ft.adapters.impl.ray.controller_actor import _FtControllerActorCls
from miles.utils.ft.adapters.stubs import StubMainJob, StubNotifier
from miles.utils.ft.controller.detectors.chain import build_detector_chain
from miles.utils.ft.controller.metrics.mini_prometheus import MiniPrometheus
from miles.utils.ft.controller.state_machines.main.models import NormalSt
from miles.utils.ft.factories.controller import build_ft_controller


def _get_training_detectors(ctrl):
    """Extract detectors from the training SubsystemConfig."""
    return ctrl._tick_loop.subsystem_configs["training"].detectors


class TestBuildFtController:
    def test_stub_platform_creates_correct_components(self) -> None:
        bundle = build_ft_controller(platform="stub", start_exporter=False, rollout_num_cells=0)
        assert isinstance(bundle.controller._main_job, StubMainJob)

    def test_stub_platform_has_full_detector_chain(self) -> None:
        bundle = build_ft_controller(platform="stub", start_exporter=False, rollout_num_cells=0)
        expected_count = len(build_detector_chain())
        assert len(_get_training_detectors(bundle.controller)) == expected_count

    def test_stub_platform_has_stub_notifier(self) -> None:
        bundle = build_ft_controller(platform="stub", start_exporter=False, rollout_num_cells=0)
        assert isinstance(bundle.controller._notifier, StubNotifier)

    def test_lark_webhook_notifier_when_url_provided(self) -> None:
        from miles.utils.ft.adapters.impl.notifiers.lark_notifier import LarkWebhookNotifier

        bundle = build_ft_controller(
            platform="stub",
            notify_webhook_url="https://hook.example.com",
            start_exporter=False,
            rollout_num_cells=0,
        )
        assert isinstance(bundle.controller._notifier, LarkWebhookNotifier)

    def test_custom_tick_interval(self) -> None:
        bundle = build_ft_controller(
            platform="stub",
            tick_interval=5.0,
            start_exporter=False,
            rollout_num_cells=0,
        )
        assert bundle.controller._tick_interval == 5.0

    def test_unknown_platform_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            build_ft_controller(platform="invalid", start_exporter=False)

    def test_unknown_backend_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            build_ft_controller(
                platform="stub",
                metric_store_backend="invalid",
                start_exporter=False,
            )

    def test_mini_backend_creates_scrape_target_manager(self) -> None:
        bundle = build_ft_controller(
            platform="stub",
            metric_store_backend="mini",
            start_exporter=False,
            rollout_num_cells=0,
        )
        assert bundle.controller._scrape_target_manager is not None

    def test_prometheus_backend_no_scrape_target_manager(self) -> None:
        bundle = build_ft_controller(
            platform="stub",
            metric_store_backend="prometheus",
            start_exporter=False,
            rollout_num_cells=0,
        )
        assert bundle.controller._scrape_target_manager is None

    def test_detector_chain_types_match(self) -> None:
        bundle = build_ft_controller(platform="stub", start_exporter=False, rollout_num_cells=0)
        expected_chain = build_detector_chain()
        actual_types = [type(d).__name__ for d in _get_training_detectors(bundle.controller)]
        expected_types = [type(d).__name__ for d in expected_chain]
        assert actual_types == expected_types

    def test_controller_exporter_registered_as_scrape_target(self) -> None:
        bundle = build_ft_controller(platform="stub", start_exporter=False, rollout_num_cells=0)
        assert isinstance(bundle.controller._metric_store, MiniPrometheus)
        assert "controller" in bundle.controller._metric_store._scrape_targets


class TestBuildPlatformComponentsK8sRay:
    def test_k8s_ray_creates_correct_types(self) -> None:
        from miles.utils.ft.factories.controller import _build_platform_components

        with (
            patch("miles.utils.ft.adapters.impl.k8s_node_manager.K8sNodeManager") as mock_k8s,
            patch("ray.job_submission.JobSubmissionClient") as mock_jsc,
            patch.dict(os.environ, {"K8S_NAMESPACE": "test-ns"}),
        ):
            mock_k8s.return_value = MagicMock()
            mock_jsc.return_value = MagicMock()

            node_mgr, main_job = _build_platform_components(
                platform="k8s-ray",
                ray_address="http://ray:8265",
                entrypoint="python train.py",
            )

        mock_k8s.assert_called_once_with(label_prefix="", namespace="test-ns")
        mock_jsc.assert_called_once_with(address="http://ray:8265")
        assert node_mgr is mock_k8s.return_value

    def test_k8s_ray_passes_ft_id_and_label_prefix(self) -> None:
        from miles.utils.ft.adapters.impl.ray.main_job import RayMainJob
        from miles.utils.ft.factories.controller import _build_platform_components

        with (
            patch("miles.utils.ft.adapters.impl.k8s_node_manager.K8sNodeManager") as mock_k8s,
            patch("ray.job_submission.JobSubmissionClient") as mock_jsc,
            patch.dict(os.environ, {"K8S_NAMESPACE": "test-ns"}),
        ):
            mock_k8s.return_value = MagicMock()
            mock_jsc.return_value = MagicMock()

            node_mgr, main_job = _build_platform_components(
                platform="k8s-ray",
                ray_address="http://ray:8265",
                entrypoint="python train.py",
                ft_id="abc",
                k8s_label_prefix="pfx",
            )

        mock_k8s.assert_called_once_with(label_prefix="pfx", namespace="test-ns")
        assert isinstance(main_job, RayMainJob)
        assert main_job._ft_id == "abc"
        assert main_job._k8s_label_prefix == "pfx"


class TestFtControllerActorProxy:
    @staticmethod
    def _make_actor_with_harness():
        harness = make_test_controller()
        actor = _FtControllerActorCls.__new__(_FtControllerActorCls)
        actor._ctrl = harness.controller
        actor._hub = harness.hub
        return actor, harness

    @pytest.mark.anyio
    async def test_register_training_rank_updates_placement(self) -> None:
        actor, harness = self._make_actor_with_harness()
        harness.controller._activate_run("test-run")

        await actor.register_training_rank(
            run_id="test-run",
            rank=0,
            world_size=2,
            node_id="node-0",
            exporter_address="http://node-0:9100",
            pid=1,
        )

        assert harness.controller.training_rank_roster.rank_placement == {0: "node-0"}

    @pytest.mark.anyio
    async def test_log_step_writes_to_mini_wandb(self) -> None:
        actor, harness = self._make_actor_with_harness()
        harness.controller._activate_run("test-run")

        await actor.register_training_rank(
            run_id="test-run",
            rank=0,
            world_size=1,
            node_id="node-0",
            exporter_address="http://node-0:9100",
            pid=1,
        )
        await actor.log_step(
            run_id="test-run",
            step=1,
            metrics={"loss": 0.5},
        )

        assert harness.mini_wandb.latest(metric_name="loss") == 0.5

    @pytest.mark.anyio
    async def test_shutdown_sets_flag(self) -> None:
        actor, harness = self._make_actor_with_harness()

        await actor.shutdown()

        assert harness.controller._shutting_down is True

    @pytest.mark.anyio
    async def test_register_rollout_creates_subsystem(self) -> None:
        actor, harness = self._make_actor_with_harness()

        fake_rollout_manager_handle = MagicMock()

        await actor.register_rollout(
            rollout_manager_handle=fake_rollout_manager_handle,
            metrics_address="http://localhost:9999",
            cell_ids=["default"],
        )

        state = harness.controller._state_machine.state
        assert isinstance(state, NormalSt)
        assert "rollout_default" in state.subsystems

    @pytest.mark.anyio
    async def test_register_rollout_adds_scrape_target(self) -> None:
        actor, harness = self._make_actor_with_harness()

        fake_rollout_manager_handle = MagicMock()
        await actor.register_rollout(
            rollout_manager_handle=fake_rollout_manager_handle,
            metrics_address="http://localhost:9999",
        )

        assert "rollout-ft-agent" in harness.controller._metric_store._scrape_targets

    @pytest.mark.anyio
    async def test_register_rollout_without_metrics_address_skips_scrape(self) -> None:
        actor, harness = self._make_actor_with_harness()

        fake_rollout_manager_handle = MagicMock()
        await actor.register_rollout(
            rollout_manager_handle=fake_rollout_manager_handle,
        )

        assert "rollout-ft-agent" not in harness.controller._metric_store._scrape_targets
