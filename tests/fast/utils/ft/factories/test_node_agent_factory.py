from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from miles.utils.ft.agents.collectors.disk import DiskCollector
from miles.utils.ft.agents.collectors.gpu import GpuCollector
from miles.utils.ft.agents.collectors.kmsg import KmsgCollector
from miles.utils.ft.agents.collectors.network import NetworkCollector
from miles.utils.ft.agents.diagnostics.executors.collector_based import CollectorBasedNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.gpu import GpuNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.nccl import NcclNodeExecutor
from miles.utils.ft.agents.diagnostics.executors.stack_trace import StackTraceNodeExecutor
from miles.utils.ft.factories.node_agent import build_all_diagnostics, build_default_collectors, build_node_agent


class TestBuildAllDiagnostics:
    def test_returns_all_executor_types(self) -> None:
        diagnostics = build_all_diagnostics(num_gpus=4)
        types = [d.diagnostic_type for d in diagnostics]
        assert types == ["stack_trace", "gpu", "nccl_simple", "nccl_pairwise", "disk", "network", "xid"]

    def test_dedicated_executors(self) -> None:
        diagnostics = build_all_diagnostics(num_gpus=4)
        assert isinstance(diagnostics[0], StackTraceNodeExecutor)
        assert isinstance(diagnostics[1], GpuNodeExecutor)
        assert isinstance(diagnostics[2], NcclNodeExecutor)
        assert isinstance(diagnostics[3], NcclNodeExecutor)

    def test_collector_based_executors_for_disk_network_xid(self) -> None:
        diagnostics = build_all_diagnostics(num_gpus=4)
        for diag in diagnostics[4:]:
            assert isinstance(diag, CollectorBasedNodeExecutor)

    def test_custom_disk_mounts_passed_through(self) -> None:
        mounts = [Path("/data"), Path("/scratch")]
        diagnostics = build_all_diagnostics(num_gpus=4, disk_mounts=mounts)
        disk_executor = diagnostics[4]
        assert isinstance(disk_executor, CollectorBasedNodeExecutor)
        assert disk_executor._collector._disk_mounts == mounts

    def test_xid_since_passed_through(self) -> None:
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        diagnostics = build_all_diagnostics(num_gpus=4, xid_since=since)
        xid_executor = diagnostics[6]
        assert isinstance(xid_executor, CollectorBasedNodeExecutor)
        assert xid_executor._collector._since == since

    def test_defaults_produce_valid_executors(self) -> None:
        diagnostics = build_all_diagnostics()
        assert len(diagnostics) == 7
        for d in diagnostics:
            assert hasattr(d, "diagnostic_type")
            assert hasattr(d, "run")


class TestBuildDefaultCollectors:
    def test_returns_four_collectors(self) -> None:
        collectors = build_default_collectors()
        assert len(collectors) == 4

    def test_returns_expected_collector_types(self) -> None:
        collectors = build_default_collectors()
        assert isinstance(collectors[0], GpuCollector)
        assert isinstance(collectors[1], KmsgCollector)
        assert isinstance(collectors[2], NetworkCollector)
        assert isinstance(collectors[3], DiskCollector)


class TestBuildNodeAgent:
    def test_uses_explicit_node_id(self) -> None:
        agent = build_node_agent(node_id="my-node")
        assert agent._node_id == "my-node"

    def test_uses_hostname_when_node_id_empty(self) -> None:
        with patch("miles.utils.ft.factories.node_agent.socket.gethostname", return_value="host-42"):
            agent = build_node_agent(node_id="")
        assert agent._node_id == "host-42"

    def test_collectors_override_replaces_defaults(self) -> None:
        custom = [GpuCollector()]
        agent = build_node_agent(node_id="n1", collectors_override=custom)
        assert len(agent._collection_loop._collectors) == 1
        assert isinstance(agent._collection_loop._collectors[0], GpuCollector)

    def test_diagnostics_override_replaces_defaults(self) -> None:
        custom = [GpuNodeExecutor()]
        agent = build_node_agent(node_id="n1", diagnostics_override=custom)
        assert list(agent._dispatcher._diagnostics.keys()) == ["gpu"]
