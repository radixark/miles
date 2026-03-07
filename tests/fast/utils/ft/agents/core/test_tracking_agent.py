"""Unit tests for FtTrackingAgent."""

from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent


class TestFtTrackingAgentLog:
    def test_log_pushes_metrics_to_controller(self) -> None:
        mock_client = MagicMock()
        agent = FtTrackingAgent(run_id="test-run-1", controller_client=mock_client)
        agent.log(metrics={"loss": 2.5, "grad_norm": 1.1}, step=10)

        mock_client.log_step.assert_called_once_with(
            run_id="test-run-1",
            step=10,
            metrics={"loss": 2.5, "grad_norm": 1.1},
        )

    def test_log_multiple_calls_accumulate(self) -> None:
        mock_client = MagicMock()
        agent = FtTrackingAgent(run_id="test-run-1", controller_client=mock_client)
        agent.log(metrics={"loss": 2.5}, step=1)
        agent.log(metrics={"loss": 1.8}, step=2)

        assert mock_client.log_step.call_count == 2

    def test_log_without_run_id_is_noop(self) -> None:
        mock_client = MagicMock()
        agent = FtTrackingAgent(run_id="", controller_client=mock_client)

        agent.log(metrics={"loss": 2.5}, step=10)

        mock_client.log_step.assert_not_called()

    def test_log_reads_run_id_from_env(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "env-run-1"}):
            agent = FtTrackingAgent()
            assert agent._run_id == "env-run-1"

    def test_log_explicit_run_id_overrides_env(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "env-run-1"}):
            agent = FtTrackingAgent(run_id="explicit-run")
            assert agent._run_id == "explicit-run"

    def test_log_exception_does_not_propagate(self) -> None:
        mock_client = MagicMock()
        mock_client.log_step.side_effect = RuntimeError("boom")

        agent = FtTrackingAgent(run_id="test-run-1", controller_client=mock_client)
        agent.log(metrics={"loss": 2.5}, step=10)

    def test_log_without_client_is_noop(self) -> None:
        agent = FtTrackingAgent(run_id="test-run-1")
        agent.log(metrics={"loss": 2.5}, step=10)


class TestTrackingUtilsIntegration:
    """Test FtTrackingAgent integration with tracking_utils.log()."""

    def test_tracking_utils_log_forwards_to_ft_agent(self) -> None:
        from miles.utils import tracking_utils

        mock_client = MagicMock()
        agent = FtTrackingAgent(run_id="test-run-1", controller_client=mock_client)
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

            mock_client.log_step.assert_called_once_with(
                run_id="test-run-1",
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
