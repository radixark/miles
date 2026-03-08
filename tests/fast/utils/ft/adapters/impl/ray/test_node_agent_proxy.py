from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.adapters.impl.ray.node_agent_proxy import RayNodeAgentProxy
from miles.utils.ft.adapters.types import NodeAgentProtocol


class TestRayNodeAgentProxy:
    def test_satisfies_node_agent_protocol(self) -> None:
        proxy = RayNodeAgentProxy(handle=MagicMock())
        assert isinstance(proxy, NodeAgentProtocol)

    @pytest.mark.anyio
    async def test_run_diagnostic_delegates_to_remote(self) -> None:
        expected = DiagnosticResult(
            diagnostic_type="gpu",
            node_id="node-0",
            passed=True,
            details="ok",
        )

        handle = MagicMock()
        future: asyncio.Future[DiagnosticResult] = asyncio.Future()
        future.set_result(expected)
        handle.run_diagnostic.remote.return_value = future

        proxy = RayNodeAgentProxy(handle=handle)
        result = await proxy.run_diagnostic(diagnostic_type="gpu", timeout_seconds=60)

        handle.run_diagnostic.remote.assert_called_once_with(
            diagnostic_type="gpu",
            timeout_seconds=60,
        )
        assert result == expected

    @pytest.mark.anyio
    async def test_kwargs_forwarded_to_remote(self) -> None:
        expected = DiagnosticResult(
            diagnostic_type="nccl_pairwise",
            node_id="node-0",
            passed=True,
            details="ok",
        )

        handle = MagicMock()
        future: asyncio.Future[DiagnosticResult] = asyncio.Future()
        future.set_result(expected)
        handle.run_diagnostic.remote.return_value = future

        proxy = RayNodeAgentProxy(handle=handle)
        result = await proxy.run_diagnostic(
            diagnostic_type="nccl_pairwise",
            timeout_seconds=120,
            master_addr="10.0.0.1",
            master_port=29500,
        )

        handle.run_diagnostic.remote.assert_called_once_with(
            diagnostic_type="nccl_pairwise",
            timeout_seconds=120,
            master_addr="10.0.0.1",
            master_port=29500,
        )
        assert result.passed is True
