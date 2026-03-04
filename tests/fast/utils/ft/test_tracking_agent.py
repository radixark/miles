"""Unit tests for FtTrackingAgent."""

from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.tracking_agent import FtTrackingAgent


class TestFtTrackingAgentLog:
    @patch("miles.utils.ft.agents.tracking_agent.FtTrackingAgent._get_controller_handle")
    def test_log_pushes_metrics_to_controller(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        agent.log(metrics={"loss": 2.5, "grad_norm": 1.1}, step=10)

        mock_controller.log_step.remote.assert_called_once_with(
            run_id="test-run-1",
            rank=0,
            step=10,
            metrics={"loss": 2.5, "grad_norm": 1.1},
        )

    @patch("miles.utils.ft.agents.tracking_agent.FtTrackingAgent._get_controller_handle")
    def test_log_multiple_calls_accumulate(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        agent.log(metrics={"loss": 2.5}, step=1)
        agent.log(metrics={"loss": 1.8}, step=2)

        assert mock_controller.log_step.remote.call_count == 2

    def test_log_without_run_id_is_noop(self) -> None:
        agent = FtTrackingAgent(rank=0, run_id="")
        agent._controller_handle = MagicMock()

        agent.log(metrics={"loss": 2.5}, step=10)

        agent._controller_handle.log_step.remote.assert_not_called()

    def test_log_reads_run_id_from_env(self) -> None:
        with patch.dict("os.environ", {"FT_TRAINING_RUN_ID": "env-run-1"}):
            agent = FtTrackingAgent(rank=0)
            assert agent._run_id == "env-run-1"

    def test_log_explicit_run_id_overrides_env(self) -> None:
        with patch.dict("os.environ", {"FT_TRAINING_RUN_ID": "env-run-1"}):
            agent = FtTrackingAgent(rank=0, run_id="explicit-run")
            assert agent._run_id == "explicit-run"

    @patch("miles.utils.ft.agents.tracking_agent.FtTrackingAgent._get_controller_handle")
    def test_log_exception_does_not_propagate(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_controller.log_step.remote.side_effect = RuntimeError("boom")
        mock_get_handle.return_value = mock_controller

        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        agent.log(metrics={"loss": 2.5}, step=10)

    def test_log_controller_unreachable_is_noop(self) -> None:
        with patch(
            "miles.utils.ft.agents.tracking_agent.FtTrackingAgent._get_controller_handle",
            return_value=None,
        ):
            agent = FtTrackingAgent(rank=0, run_id="test-run-1")
            agent.log(metrics={"loss": 2.5}, step=10)


class TestFtTrackingAgentControllerHandle:
    def test_get_controller_handle_caches_result(self) -> None:
        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        mock_handle = MagicMock()
        agent._controller_handle = mock_handle

        with patch("ray.get_actor") as mock_get_actor:
            result = agent._get_controller_handle()

            assert result is mock_handle
            mock_get_actor.assert_not_called()

    def test_get_controller_handle_negative_cache(self) -> None:
        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        agent._controller_lookup_failed = True

        with patch("ray.get_actor") as mock_get_actor:
            result = agent._get_controller_handle()

            assert result is None
            mock_get_actor.assert_not_called()

    def test_reset_controller_handle(self) -> None:
        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        agent._controller_handle = MagicMock()
        agent._controller_lookup_failed = True

        agent._reset_controller_handle()

        assert agent._controller_handle is None
        assert agent._controller_lookup_failed is False


class TestTrackingUtilsIntegration:
    """Test FtTrackingAgent integration with tracking_utils.log()."""

    @patch("miles.utils.ft.agents.tracking_agent.FtTrackingAgent._get_controller_handle")
    def test_tracking_utils_log_forwards_to_ft_agent(
        self, mock_get_handle: MagicMock
    ) -> None:
        from miles.utils import tracking_utils

        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        agent = FtTrackingAgent(rank=0, run_id="test-run-1")
        tracking_utils.set_ft_tracking_agent(agent)
        try:
            args = MagicMock()
            args.use_wandb = False
            args.use_tensorboard = False

            tracking_utils.log(
                args,
                {"train/loss": 2.5, "train/grad_norm": 1.1, "train/step": 42},
                step_key="train/step",
            )

            mock_controller.log_step.remote.assert_called_once_with(
                run_id="test-run-1",
                rank=0,
                step=42,
                metrics={"train/loss": 2.5, "train/grad_norm": 1.1},
            )
        finally:
            tracking_utils.set_ft_tracking_agent(None)

    def test_tracking_utils_log_without_agent_is_noop(self) -> None:
        from miles.utils import tracking_utils

        tracking_utils.set_ft_tracking_agent(None)

        args = MagicMock()
        args.use_wandb = False
        args.use_tensorboard = False

        tracking_utils.log(
            args,
            {"train/loss": 2.5, "train/step": 42},
            step_key="train/step",
        )
