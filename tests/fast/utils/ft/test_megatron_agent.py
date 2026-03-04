"""Unit tests for FtMegatronAgent."""

from collections.abc import AsyncIterator

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
