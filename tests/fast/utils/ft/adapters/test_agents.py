"""Protocol compliance tests for NodeAgentProtocol and NodeExecutorProtocol."""

from __future__ import annotations

from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl_simple import NcclSimpleNodeExecutor
from miles.utils.ft.agents.diagnostics.dispatcher import NodeDiagnosticDispatcher
from miles.utils.ft.agents.diagnostics.executors.stack_trace import StackTraceNodeExecutor
from miles.utils.ft.adapters.types import NodeExecutorProtocol, NodeAgentProtocol


class TestNodeAgentProtocolCompliance:
    def test_ft_node_agent_satisfies_protocol(self) -> None:
        agent = FtNodeAgent(node_id="test-node")
        assert isinstance(agent, NodeAgentProtocol)
        agent._exporter.shutdown()

    def test_diagnostic_runner_satisfies_protocol(self) -> None:
        runner = NodeDiagnosticDispatcher(node_id="test-node")
        assert isinstance(runner, NodeAgentProtocol)

    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def run_diagnostic(
                self,
                diagnostic_type: str,
                timeout_seconds: int = 120,
                **kwargs: object,
            ) -> object:
                return None

        assert isinstance(_Conforming(), NodeAgentProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _Empty:
            pass

        assert not isinstance(_Empty(), NodeAgentProtocol)


class TestNodeExecutorProtocolCompliance:
    def test_gpu_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(GpuNodeExecutor(), NodeExecutorProtocol)

    def test_stack_trace_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(StackTraceNodeExecutor(), NodeExecutorProtocol)

    def test_nccl_simple_diagnostic_satisfies_protocol(self) -> None:
        assert isinstance(NcclSimpleNodeExecutor(), NodeExecutorProtocol)

    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            diagnostic_type = "test"

            async def run(self, node_id: str, timeout_seconds: int = 120) -> object:
                return None

        assert isinstance(_Conforming(), NodeExecutorProtocol)

    def test_missing_run_method_fails_isinstance(self) -> None:
        class _MissingRun:
            diagnostic_type = "test"

        assert not isinstance(_MissingRun(), NodeExecutorProtocol)
