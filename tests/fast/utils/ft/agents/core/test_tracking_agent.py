"""Unit tests for FtTrackingAgent."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.core.tracking_agent import FtTrackingAgent
from miles.utils.ft.utils.graceful_degrade import FaultInjectionError


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


class TestCheckExceptionInjection:
    def test_raises_fault_injection_error_when_trigger_file_exists(self, tmp_path: Path) -> None:
        trigger_file = tmp_path / "inject_fault"
        trigger_file.touch()

        env = {"MILES_FT_EXCEPTION_INJECT_PATH": str(trigger_file)}
        with patch.dict("os.environ", env, clear=False):
            agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock())
            with pytest.raises(FaultInjectionError):
                agent.log(metrics={"loss": 1.0}, step=1)

    def test_trigger_file_deleted_after_injection(self, tmp_path: Path) -> None:
        trigger_file = tmp_path / "inject_fault"
        trigger_file.touch()

        env = {"MILES_FT_EXCEPTION_INJECT_PATH": str(trigger_file)}
        with patch.dict("os.environ", env, clear=False):
            agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock())
            with pytest.raises(FaultInjectionError):
                agent.log(metrics={"loss": 1.0}, step=1)

        assert not trigger_file.exists()

    def test_noop_when_trigger_file_absent(self, tmp_path: Path) -> None:
        trigger_file = tmp_path / "inject_fault"

        env = {"MILES_FT_EXCEPTION_INJECT_PATH": str(trigger_file)}
        with patch.dict("os.environ", env, clear=False):
            mock_client = MagicMock()
            agent = FtTrackingAgent(run_id="r1", controller_client=mock_client)
            agent.log(metrics={"loss": 1.0}, step=1)

        mock_client.log_step.assert_called_once()

    def test_noop_when_env_var_not_set(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            # Ensure the env var is absent
            import os

            os.environ.pop("MILES_FT_EXCEPTION_INJECT_PATH", None)
            os.environ.pop("MILES_FT_EXCEPTION_INJECT_DIR", None)

            mock_client = MagicMock()
            agent = FtTrackingAgent(run_id="r1", controller_client=mock_client)
            agent.log(metrics={"loss": 1.0}, step=1)

        mock_client.log_step.assert_called_once()


class TestPerRankExceptionInjection:
    """Tests for per-rank broadcast-style exception injection.

    The original single-path design had a TOCTOU race: multiple processes
    sharing the same flag file would race on exists()+unlink(), causing
    only the first checker to observe the injection while others missed it.
    The per-rank directory design gives each rank its own flag file so all
    target processes independently consume their own injection signal."""

    def test_per_rank_flag_path_uses_rank_specific_file(self, tmp_path: Path) -> None:
        inject_dir = tmp_path / "ft_exc"
        inject_dir.mkdir()
        flag = inject_dir / "exception.rank-3"
        flag.touch()

        env = {"MILES_FT_EXCEPTION_INJECT_DIR": str(inject_dir)}
        with patch.dict("os.environ", env, clear=False):
            agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock(), rank=3)
            with pytest.raises(FaultInjectionError, match="exception.rank-3"):
                agent.log(metrics={"loss": 1.0}, step=1)

        assert not flag.exists()

    def test_different_ranks_get_different_flag_paths(self, tmp_path: Path) -> None:
        inject_dir = tmp_path / "ft_exc"
        inject_dir.mkdir()

        env = {"MILES_FT_EXCEPTION_INJECT_DIR": str(inject_dir)}
        with patch.dict("os.environ", env, clear=False):
            agent_0 = FtTrackingAgent(run_id="r1", rank=0)
            agent_1 = FtTrackingAgent(run_id="r1", rank=1)

        assert agent_0._exception_inject_path != agent_1._exception_inject_path
        assert "rank-0" in str(agent_0._exception_inject_path)
        assert "rank-1" in str(agent_1._exception_inject_path)

    def test_broadcast_injection_hits_all_target_ranks(self, tmp_path: Path) -> None:
        """Multiple ranks each consume their own flag file independently,
        so a broadcast injection triggers all targets without any race."""
        inject_dir = tmp_path / "ft_exc"
        inject_dir.mkdir()

        from miles.utils.ft.utils.env import build_exception_inject_flag_path

        for rank in [0, 1, 2]:
            build_exception_inject_flag_path(inject_dir, rank=rank).touch()

        env = {"MILES_FT_EXCEPTION_INJECT_DIR": str(inject_dir)}
        triggered_ranks: list[int] = []

        with patch.dict("os.environ", env, clear=False):
            for rank in [0, 1, 2]:
                agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock(), rank=rank)
                try:
                    agent.log(metrics={"loss": 1.0}, step=1)
                except FaultInjectionError:
                    triggered_ranks.append(rank)

        assert triggered_ranks == [0, 1, 2]

    def test_dir_env_takes_precedence_over_path_env_when_rank_provided(self, tmp_path: Path) -> None:
        """When both MILES_FT_EXCEPTION_INJECT_DIR and
        MILES_FT_EXCEPTION_INJECT_PATH are set and rank is provided,
        the dir-based per-rank path should be used."""
        inject_dir = tmp_path / "ft_exc"
        inject_dir.mkdir()
        old_path = tmp_path / "old_flag"
        old_path.touch()

        flag = inject_dir / "exception.rank-0"
        flag.touch()

        env = {
            "MILES_FT_EXCEPTION_INJECT_DIR": str(inject_dir),
            "MILES_FT_EXCEPTION_INJECT_PATH": str(old_path),
        }
        with patch.dict("os.environ", env, clear=False):
            agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock(), rank=0)
            with pytest.raises(FaultInjectionError, match="exception.rank-0"):
                agent.log(metrics={"loss": 1.0}, step=1)

        assert old_path.exists()

    def test_falls_back_to_legacy_path_when_no_rank(self, tmp_path: Path) -> None:
        """When rank is not provided, falls back to the legacy single-path."""
        trigger_file = tmp_path / "inject_fault"
        trigger_file.touch()

        env = {"MILES_FT_EXCEPTION_INJECT_PATH": str(trigger_file)}
        with patch.dict("os.environ", env, clear=False):
            agent = FtTrackingAgent(run_id="r1", controller_client=MagicMock())
            with pytest.raises(FaultInjectionError):
                agent.log(metrics={"loss": 1.0}, step=1)
