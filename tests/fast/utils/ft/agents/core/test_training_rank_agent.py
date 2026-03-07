"""Unit tests for FtTrainingRankAgent.

FtTrainingRankAgent delegates metric exposition to TrainingRankMetricExporter
and is responsible for rank registration with FtController, plus the
maybe_create factory.  Exporter-level tests live in
test_training_rank_metric_exporter.py.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.agents.core.training_rank_agent import FtTrainingRankAgent


@contextmanager
def _registered_agent(
    mock_get_handle: MagicMock, **agent_kwargs: Any
) -> Iterator[tuple[FtTrainingRankAgent, dict[str, Any]]]:
    mock_controller = MagicMock()
    mock_ray_get = MagicMock()
    mock_get_handle.return_value = mock_controller
    defaults: dict[str, Any] = {"rank": 0, "world_size": 4}
    defaults.update(agent_kwargs)
    with patch.dict(
        "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
    ), patch("ray.get", mock_ray_get):
        agent = FtTrainingRankAgent(**defaults)
        try:
            call_kwargs = mock_controller.register_training_rank.remote.call_args[1]
            yield agent, call_kwargs
        finally:
            agent.shutdown()


class TestFtTrainingRankAgentRegisterRank:
    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_calls_controller(
        self, mock_get_handle: MagicMock
    ) -> None:
        with _registered_agent(mock_get_handle) as (agent, call_kwargs):
            mock_get_handle.return_value.register_training_rank.remote.assert_called_once()
            assert call_kwargs["run_id"] == "test-run-1"
            assert call_kwargs["rank"] == 0
            assert call_kwargs["world_size"] == 4

    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_retries_on_failure(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        call_count = 0

        def ray_get_side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated failure")
            return None

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", side_effect=ray_get_side_effect), patch(
            "time.sleep"
        ):
            agent = FtTrainingRankAgent(rank=0, world_size=4)
            try:
                assert call_count == 3
                assert mock_controller.register_training_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_all_attempts_fail_no_exception(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch(
            "ray.get", side_effect=RuntimeError("always fails")
        ), patch("time.sleep"):
            agent = FtTrainingRankAgent(rank=2, world_size=4)
            try:
                assert mock_controller.register_training_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    def test_register_training_rank_skipped_without_run_id(self) -> None:
        agent = FtTrainingRankAgent(rank=0, world_size=4)
        try:
            assert agent._run_id == ""
        finally:
            agent.shutdown()

    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_skipped_when_controller_unavailable(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_get_handle.return_value = None

        with patch.dict("os.environ", {"MILES_FT_TRAINING_RUN_ID": "test-run-1"}):
            agent = FtTrainingRankAgent(rank=0, world_size=4)
            try:
                assert agent._run_id == "test-run-1"
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_asserts_node_id_and_exporter_address(
        self, mock_get_handle: MagicMock
    ) -> None:
        with _registered_agent(mock_get_handle) as (agent, call_kwargs):
            assert call_kwargs["node_id"] == agent._node_id
            assert call_kwargs["exporter_address"] == agent.get_exporter_address()

    @patch("miles.utils.ft.agents.core.training_rank_agent.get_controller_handle")
    def test_register_training_rank_includes_pid(
        self, mock_get_handle: MagicMock
    ) -> None:
        import os as _os

        with _registered_agent(mock_get_handle) as (agent, call_kwargs):
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
        with patch.object(
            FtTrainingRankAgent, "__init__", side_effect=RuntimeError("init failed")
        ):
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
        agent = FtTrainingRankAgent(rank=0, world_size=4)
        try:
            agent.step()
            assert agent._metric_exporter._heartbeat_counter == 1
        finally:
            agent.shutdown()
