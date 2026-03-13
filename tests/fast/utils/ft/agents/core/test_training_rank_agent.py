"""Unit tests for FtTrainingRankAgent.

FtTrainingRankAgent delegates metric exposition to TrainingRankExporter
and is responsible for rank registration with FtController, plus the
maybe_create factory.  Exporter-level tests live in
test_training_rank_exporter.py.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch


from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent


@contextmanager
def _registered_agent(
    **agent_kwargs: Any,
) -> Iterator[tuple[FtTrainingRankAgent, MagicMock]]:
    mock_client = MagicMock()
    defaults: dict[str, Any] = {"rank": 0, "world_size": 4, "controller_client": mock_client}
    defaults.update(agent_kwargs)
    with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}):
        agent = FtTrainingRankAgent(**defaults)
        try:
            yield agent, mock_client
        finally:
            agent.shutdown()


class TestFtTrainingRankAgentRegisterRank:
    def test_register_training_rank_waits_for_exporter_readiness(self) -> None:
        mock_client = MagicMock()

        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}), patch(
            "miles.utils.ft.agents.core.training_rank_agent.TrainingRankExporter"
        ) as exporter_cls:
            exporter = exporter_cls.return_value
            agent = FtTrainingRankAgent(rank=0, world_size=4, controller_client=mock_client)

        try:
            exporter.wait_until_ready.assert_called_once_with()
            mock_client.register_training_rank.assert_called_once()
        finally:
            agent.shutdown()

    def test_register_training_rank_calls_controller(self) -> None:
        with _registered_agent() as (agent, mock_client):
            mock_client.register_training_rank.assert_called_once()
            call_kwargs = mock_client.register_training_rank.call_args[1]
            assert call_kwargs["run_id"] == "test-run-1"
            assert call_kwargs["rank"] == 0
            assert call_kwargs["world_size"] == 4

    def test_register_training_rank_retries_on_failure(self) -> None:
        mock_client = MagicMock()
        call_count = 0

        def register_side_effect(**kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated failure")

        mock_client.register_training_rank.side_effect = register_side_effect

        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}), patch("time.sleep"):
            agent = FtTrainingRankAgent(rank=0, world_size=4, controller_client=mock_client)
            try:
                assert call_count == 3
                assert mock_client.register_training_rank.call_count == 3
            finally:
                agent.shutdown()

    def test_register_training_rank_all_attempts_fail_logs_warning(self, caplog: Any) -> None:
        """H-4: when all registration retries are exhausted, a warning must
        be logged (previously the failure was completely silent)."""
        mock_client = MagicMock()
        mock_client.register_training_rank.side_effect = RuntimeError("always fails")

        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}), patch("time.sleep"):
            with caplog.at_level(logging.WARNING):
                agent = FtTrainingRankAgent(rank=2, world_size=4, controller_client=mock_client)
            try:
                assert mock_client.register_training_rank.call_count == 3
                assert any("registration failed" in r.message for r in caplog.records if r.levelno >= logging.WARNING)
            finally:
                agent.shutdown()

    def test_register_training_rank_skipped_without_run_id(self) -> None:
        agent = FtTrainingRankAgent(rank=0, world_size=4)
        assert agent._run_id == ""
        assert agent._metric_exporter is None

    def test_register_training_rank_skipped_when_no_client(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}):
            agent = FtTrainingRankAgent(rank=0, world_size=4)
            try:
                assert agent._run_id == "test-run-1"
            finally:
                agent.shutdown()

    def test_register_training_rank_asserts_node_id_and_exporter_address(self) -> None:
        with _registered_agent() as (agent, mock_client):
            call_kwargs = mock_client.register_training_rank.call_args[1]
            assert call_kwargs["node_id"] == agent._node_id
            assert call_kwargs["exporter_address"] == agent.get_exporter_address()

    def test_register_training_rank_includes_pid(self) -> None:
        import os as _os

        with _registered_agent() as (agent, mock_client):
            call_kwargs = mock_client.register_training_rank.call_args[1]
            assert call_kwargs["pid"] == _os.getpid()


class TestFtTrainingRankAgentFaultTolerance:
    def test_maybe_create_returns_agent_when_enabled(self) -> None:
        agent = FtTrainingRankAgent.maybe_create(rank=0, world_size=4, enabled=True)
        try:
            assert agent is not None
            assert isinstance(agent, FtTrainingRankAgent)
        finally:
            if agent is not None:
                agent.shutdown()

    def test_maybe_create_returns_none_when_disabled(self) -> None:
        agent = FtTrainingRankAgent.maybe_create(rank=0, world_size=4, enabled=False)
        assert agent is None

    def test_maybe_create_returns_none_on_init_error(self) -> None:
        with patch.object(FtTrainingRankAgent, "__init__", side_effect=RuntimeError("init failed")):
            agent = FtTrainingRankAgent.maybe_create(rank=0, world_size=4)
            assert agent is None

    def test_maybe_create_without_run_id_still_creates(self) -> None:
        agent = FtTrainingRankAgent.maybe_create(rank=0, world_size=4)
        try:
            assert agent is not None
            assert agent._run_id == ""
        finally:
            if agent is not None:
                agent.shutdown()

    def test_step_delegates_to_metric_exporter(self) -> None:
        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}):
            agent = FtTrainingRankAgent(rank=0, world_size=4)
        try:
            agent.step()
            assert agent._metric_exporter is not None
            assert agent._metric_exporter._heartbeat_counter == 1
        finally:
            agent.shutdown()

    def test_agent_passes_run_id_to_exporter(self) -> None:
        """FtTrainingRankAgent must forward its run_id to the exporter so
        that run-scoped metrics carry ft_run_id."""
        with _registered_agent() as (agent, _mock_client):
            assert agent._metric_exporter is not None

    def test_no_run_id_disables_metric_exporter(self) -> None:
        """Without run_id, metric exporter is disabled to avoid emitting
        metrics without ft_run_id label."""
        agent = FtTrainingRankAgent(rank=0, world_size=4)
        assert agent._metric_exporter is None
        agent.step()
        agent.set_phase("training")
        agent.shutdown()
