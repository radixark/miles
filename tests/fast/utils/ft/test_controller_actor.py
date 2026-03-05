from __future__ import annotations

from unittest.mock import patch

import pytest

from miles.utils.ft.controller.detectors import build_detector_chain
from miles.utils.ft.platform.controller_actor import (
    _FtControllerActorCls,
    build_ft_controller,
)
from miles.utils.ft.platform.stubs import StubNodeManager, StubNotifier, StubTrainingJob
from tests.fast.utils.ft.conftest import make_test_controller

from unittest.mock import MagicMock


class TestBuildFtController:
    def test_stub_platform_creates_correct_components(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        assert isinstance(ctrl._node_manager, StubNodeManager)
        assert isinstance(ctrl._training_job, StubTrainingJob)

    def test_stub_platform_has_full_detector_chain(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        expected_count = len(build_detector_chain())
        assert len(ctrl._detectors) == expected_count

    def test_stub_platform_has_stub_notifier(self) -> None:
        ctrl = build_ft_controller(platform="stub", start_exporter=False)
        assert isinstance(ctrl._notifier, StubNotifier)

    def test_lark_webhook_notifier_when_env_set(self) -> None:
        from miles.utils.ft.platform.lark_notifier import LarkWebhookNotifier

        with patch.dict("os.environ", {"FT_LARK_WEBHOOK_URL": "https://hook.example.com"}):
            ctrl = build_ft_controller(platform="stub", start_exporter=False)
        assert isinstance(ctrl._notifier, LarkWebhookNotifier)

    def test_custom_tick_interval(self) -> None:
        ctrl = build_ft_controller(
            platform="stub", tick_interval=5.0, start_exporter=False,
        )
        assert ctrl._tick_interval == 5.0

    def test_unknown_platform_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown platform"):
            build_ft_controller(platform="invalid", start_exporter=False)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric-store-backend"):
            build_ft_controller(
                platform="stub",
                metric_store_backend="invalid",
                start_exporter=False,
            )

    def test_mini_backend_creates_scrape_target_manager(self) -> None:
        ctrl = build_ft_controller(
            platform="stub",
            metric_store_backend="mini",
            start_exporter=False,
        )
        assert ctrl._scrape_target_manager is not None

    def test_prometheus_backend_no_scrape_target_manager(self) -> None:
        ctrl = build_ft_controller(
            platform="stub",
            metric_store_backend="prometheus",
            start_exporter=False,
        )
        assert ctrl._scrape_target_manager is None


class TestBuildPlatformComponentsK8sRay:
    def test_k8s_ray_creates_correct_types(self) -> None:
        from miles.utils.ft.platform.controller_actor import _build_platform_components

        with (
            patch("miles.utils.ft.platform.controller_actor.K8sNodeManager") as mock_k8s,
            patch("miles.utils.ft.platform.ray_training_job.JobSubmissionClient") as mock_jsc,
        ):
            mock_k8s.return_value = MagicMock()
            mock_jsc.return_value = MagicMock()

            node_mgr, training_job = _build_platform_components(
                platform="k8s-ray",
                ray_address="http://ray:8265",
                entrypoint="python train.py",
            )

        mock_k8s.assert_called_once()
        mock_jsc.assert_called_once_with(address="http://ray:8265")
        assert node_mgr is mock_k8s.return_value


class TestFtControllerActorProxy:
    @staticmethod
    def _make_actor_with_harness():
        harness = make_test_controller()
        actor = _FtControllerActorCls.__new__(_FtControllerActorCls)
        actor._ctrl = harness.controller
        return actor, harness

    @pytest.mark.asyncio
    async def test_register_rank_updates_placement(self) -> None:
        actor, harness = self._make_actor_with_harness()

        await actor.register_rank(
            run_id="test-run", rank=0, world_size=2,
            node_id="node-0", exporter_address="http://node-0:9100",
        )

        assert harness.controller._rank_placement == {0: "node-0"}

    @pytest.mark.asyncio
    async def test_log_step_writes_to_mini_wandb(self) -> None:
        actor, harness = self._make_actor_with_harness()

        await actor.register_rank(
            run_id="test-run", rank=0, world_size=1,
            node_id="node-0", exporter_address="http://node-0:9100",
        )
        await actor.log_step(
            run_id="test-run", rank=0, step=1, metrics={"loss": 0.5},
        )

        assert harness.mini_wandb.latest(metric_name="loss", rank=0) == 0.5

    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self) -> None:
        actor, harness = self._make_actor_with_harness()

        await actor.shutdown()

        assert harness.controller._shutting_down is True
