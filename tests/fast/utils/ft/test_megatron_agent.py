"""Unit tests for FtMegatronAgent.

FtMegatronAgent is responsible only for heartbeat gauges (iteration + phase)
exposed via a Prometheus HTTP exporter, and rank registration with FtController.
Training metrics are forwarded separately by FtTrackingAgent via tracking_utils.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest

from miles.utils.ft.agents.megatron_agent import FtMegatronAgent


@pytest.fixture()
def agent() -> Iterator[FtMegatronAgent]:
    agent = FtMegatronAgent(rank=0, world_size=4)
    yield agent
    agent.shutdown()


class TestFtMegatronAgentExporter:
    @pytest.mark.asyncio()
    async def test_exporter_returns_prometheus_format(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.asyncio()
    async def test_exporter_address_has_port(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.asyncio()
    async def test_initial_gauge_values(
        self, agent: FtMegatronAgent
    ) -> None:
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "training_iteration" in text
        assert "training_phase" in text
        assert 'rank="0"' in text


class TestFtMegatronAgentStep:
    @pytest.mark.asyncio()
    async def test_step_updates_iteration_gauge(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.step(iteration=42)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "training_iteration" in text
        assert "42.0" in text

    @pytest.mark.asyncio()
    async def test_step_updates_phase_gauge(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.step(iteration=1, phase="checkpoint_saving")

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert "2.0" in text

    def test_step_does_not_interact_with_controller(
        self, agent: FtMegatronAgent
    ) -> None:
        agent._controller_handle = MagicMock()
        agent.step(iteration=10)
        agent._controller_handle.log_step.remote.assert_not_called()


class TestFtMegatronAgentRegisterRank:
    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_calls_controller(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_ray_get = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", mock_ray_get):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                mock_controller.register_rank.remote.assert_called_once()
                call_kwargs = mock_controller.register_rank.remote.call_args[1]
                assert call_kwargs["run_id"] == "test-run-1"
                assert call_kwargs["rank"] == 0
                assert call_kwargs["world_size"] == 4
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_retries_on_failure(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        call_count = 0

        def ray_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated failure")
            return None

        with patch.dict(
            "os.environ", {"FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", side_effect=ray_get_side_effect), patch(
            "time.sleep"
        ):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                assert call_count == 3
                assert mock_controller.register_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_all_attempts_fail_no_exception(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch(
            "ray.get", side_effect=RuntimeError("always fails")
        ), patch("time.sleep"):
            agent = FtMegatronAgent(rank=2, world_size=4)
            try:
                assert mock_controller.register_rank.remote.call_count == 3
            finally:
                agent.shutdown()

    def test_register_rank_skipped_without_run_id(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            assert agent._run_id == ""
        finally:
            agent.shutdown()

    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_skipped_when_controller_unavailable(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_get_handle.return_value = None

        with patch.dict("os.environ", {"FT_TRAINING_RUN_ID": "test-run-1"}):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                assert agent._run_id == "test-run-1"
            finally:
                agent.shutdown()

    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_register_rank_asserts_node_id_and_exporter_address(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_ray_get = MagicMock()
        mock_get_handle.return_value = mock_controller

        with patch.dict(
            "os.environ", {"FT_TRAINING_RUN_ID": "test-run-1"}
        ), patch("ray.get", mock_ray_get):
            agent = FtMegatronAgent(rank=0, world_size=4)
            try:
                call_kwargs = mock_controller.register_rank.remote.call_args[1]
                assert call_kwargs["node_id"] == agent._node_id
                assert call_kwargs["exporter_address"] == agent.get_exporter_address()
            finally:
                agent.shutdown()


class TestFtMegatronAgentFaultTolerance:
    def test_maybe_create_returns_agent_when_enabled(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4, enabled=True)
        try:
            assert agent is not None
            assert isinstance(agent, FtMegatronAgent)
        finally:
            if agent is not None:
                agent.shutdown()

    def test_maybe_create_returns_none_when_disabled(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4, enabled=False)
        assert agent is None

    def test_maybe_create_returns_none_on_init_error(self) -> None:
        with patch.object(
            FtMegatronAgent, "__init__", side_effect=RuntimeError("init failed")
        ):
            agent = FtMegatronAgent.maybe_create(rank=0, world_size=4)
            assert agent is None

    def test_maybe_create_without_run_id_still_creates(self) -> None:
        agent = FtMegatronAgent.maybe_create(rank=0, world_size=4)
        try:
            assert agent is not None
            assert agent._run_id == ""
        finally:
            if agent is not None:
                agent.shutdown()

    def test_step_exception_does_not_propagate(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            with patch.object(
                agent, "_iteration_child", **{"set.side_effect": RuntimeError("boom")}
            ):
                agent.step(iteration=1)
        finally:
            agent.shutdown()

    def test_reset_controller_handle(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            agent._controller_handle = MagicMock()
            agent._controller_lookup_failed = True

            agent._reset_controller_handle()

            assert agent._controller_handle is None
            assert agent._controller_lookup_failed is False
        finally:
            agent.shutdown()

    def test_get_controller_handle_caches_result(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            mock_handle = MagicMock()
            agent._controller_handle = mock_handle

            with patch("ray.get_actor") as mock_get_actor:
                result = agent._get_controller_handle()

                assert result is mock_handle
                mock_get_actor.assert_not_called()
        finally:
            agent.shutdown()

    def test_get_controller_handle_negative_cache(self) -> None:
        agent = FtMegatronAgent(rank=0, world_size=4)
        try:
            agent._controller_lookup_failed = True

            with patch("ray.get_actor") as mock_get_actor:
                result = agent._get_controller_handle()

                assert result is None
                mock_get_actor.assert_not_called()
        finally:
            agent.shutdown()
