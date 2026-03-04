"""Unit tests for FtMegatronAgent."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from miles.utils.ft.agents.megatron_agent import FtMegatronAgent


class TestFtMegatronAgentExporter:
    @pytest.fixture()
    def agent(self) -> FtMegatronAgent:
        agent = FtMegatronAgent(rank=0, world_size=4)
        yield agent
        agent._httpd.shutdown()
        agent._httpd.server_close()

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
    @pytest.fixture()
    def agent(self) -> FtMegatronAgent:
        agent = FtMegatronAgent(rank=0, world_size=4)
        yield agent
        agent._httpd.shutdown()
        agent._httpd.server_close()

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

    def test_step_with_nan_loss_does_not_raise(
        self, agent: FtMegatronAgent
    ) -> None:
        agent.step(iteration=1, loss=float("nan"))

    @patch("miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle")
    def test_step_pushes_metrics_to_controller(
        self, mock_get_handle: MagicMock
    ) -> None:
        mock_controller = MagicMock()
        mock_get_handle.return_value = mock_controller

        agent = FtMegatronAgent(rank=0, world_size=4)
        agent._run_id = "test-run-1"
        try:
            agent.step(iteration=10, loss=2.5, grad_norm=1.1)

            mock_controller.log_step.remote.assert_called_once_with(
                run_id="test-run-1",
                rank=0,
                step=10,
                metrics={"loss": 2.5, "grad_norm": 1.1},
            )
        finally:
            agent._httpd.shutdown()
            agent._httpd.server_close()

    def test_step_without_metrics_does_not_push(
        self, agent: FtMegatronAgent
    ) -> None:
        agent._run_id = "test-run-1"
        agent._controller_handle = MagicMock()

        agent.step(iteration=10)

        agent._controller_handle.log_step.remote.assert_not_called()

    def test_step_without_run_id_does_not_push(
        self, agent: FtMegatronAgent
    ) -> None:
        agent._run_id = ""
        agent._controller_handle = MagicMock()

        agent.step(iteration=10, loss=2.5)

        agent._controller_handle.log_step.remote.assert_not_called()

    def test_controller_unreachable_does_not_raise(self) -> None:
        with patch(
            "miles.utils.ft.agents.megatron_agent.FtMegatronAgent._get_controller_handle",
            return_value=None,
        ):
            agent = FtMegatronAgent(rank=0, world_size=4)
            agent._run_id = "test-run-1"
            try:
                agent.step(iteration=10, loss=2.5)
            finally:
                agent._httpd.shutdown()
                agent._httpd.server_close()
