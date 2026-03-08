import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from tests.fast.utils.ft.conftest import SlowDiagnostic, StubDiagnostic, TestCollector
from tests.fast.utils.ft.utils import FailingCloseCollector, FailingCollector

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.stub import StubCollector
from miles.utils.ft.agents.core.node_agent import FtNodeAgent
from miles.utils.ft.models.diagnostics import DiagnosticResult, UnknownDiagnosticError
from miles.utils.ft.models.metrics import CollectorOutput, GaugeSample


class _CountingCollector(BaseCollector):
    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.call_count = 0
        self.closed = False

    def _collect_sync(self) -> list[GaugeSample]:
        self.call_count += 1
        return [GaugeSample(name="count", labels={}, value=float(self.call_count))]

    async def collect(self) -> CollectorOutput:
        self.call_count += 1
        return CollectorOutput(
            metrics=[
                GaugeSample(name="count", labels={}, value=float(self.call_count)),
            ]
        )

    async def close(self) -> None:
        self.closed = True


MakeNodeAgent = Callable[..., FtNodeAgent]


@pytest.fixture
async def make_node_agent() -> AsyncIterator[MakeNodeAgent]:
    agents: list[FtNodeAgent] = []

    def factory(**kwargs: Any) -> FtNodeAgent:
        kwargs.setdefault("node_id", "test-node")
        agent = FtNodeAgent(**kwargs)
        agents.append(agent)
        return agent

    yield factory
    for agent in agents:
        await agent.stop()


class TestFtNodeAgentExporter:
    @pytest.mark.anyio
    async def test_exporter_returns_prometheus_format(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-0")
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.anyio
    async def test_exporter_address_has_port(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-0")
        address = agent.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.anyio
    async def test_stub_collector_no_custom_metrics(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(
            node_id="test-node-stub",
            collectors=[StubCollector()],
        )
        agent._exporter.update_metrics([])

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "miles_ft_" not in response.text

    @pytest.mark.anyio
    async def test_update_exporter_creates_gauges(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-gauge")
        metrics = [
            GaugeSample(
                name="gpu_temperature_celsius",
                labels={"gpu": "0"},
                value=75.0,
            ),
            GaugeSample(
                name="gpu_temperature_celsius",
                labels={"gpu": "1"},
                value=80.0,
            ),
        ]
        agent._exporter.update_metrics(metrics)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        text = response.text
        assert 'gpu_temperature_celsius{gpu="0"}' in text
        assert 'gpu_temperature_celsius{gpu="1"}' in text
        assert "75.0" in text
        assert "80.0" in text

    @pytest.mark.anyio
    async def test_update_exporter_unlabeled_metric(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-unlabeled")
        metrics = [
            GaugeSample(name="uptime_seconds", labels={}, value=123.0),
        ]
        agent._exporter.update_metrics(metrics)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert "uptime_seconds" in response.text
        assert "123.0" in response.text

    @pytest.mark.anyio
    async def test_update_exporter_overwrites_value(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-overwrite")
        agent._exporter.update_metrics(
            [
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=60.0,
                ),
            ]
        )
        agent._exporter.update_metrics(
            [
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=90.0,
                ),
            ]
        )

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert "90.0" in response.text


class TestFtNodeAgentCollectionLoop:
    @pytest.mark.anyio
    async def test_collection_loop_updates_exporter(self, make_node_agent: MakeNodeAgent) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=65.0,
                ),
            ],
            collect_interval=0.05,
        )
        agent = make_node_agent(
            node_id="test-node-loop",
            collectors=[test_collector],
        )
        await agent.start()
        await asyncio.sleep(0.3)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert 'gpu_temperature_celsius{gpu="0"}' in response.text
        assert "65.0" in response.text

    @pytest.mark.anyio
    async def test_stop_cancels_tasks(self) -> None:
        test_collector = TestCollector(
            metrics=[GaugeSample(name="dummy", labels={}, value=1.0)],
            collect_interval=0.05,
        )
        agent = FtNodeAgent(
            node_id="test-node-stop",
            collectors=[test_collector],
        )
        await agent.start()

        assert len(agent._collection_loop.tasks) == 1
        await agent.stop()
        assert len(agent._collection_loop.tasks) == 0

    @pytest.mark.anyio
    async def test_failing_collector_does_not_crash_loop(self, make_node_agent: MakeNodeAgent) -> None:
        good_collector = TestCollector(
            metrics=[GaugeSample(name="good_metric", labels={}, value=42.0)],
            collect_interval=0.05,
        )
        failing = FailingCollector(collect_interval=0.05)
        agent = make_node_agent(
            node_id="test-node-fail",
            collectors=[failing, good_collector],
        )
        await agent.start()
        await asyncio.sleep(0.3)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert "good_metric" in response.text
        assert "42.0" in response.text

    @pytest.mark.anyio
    async def test_all_collectors_failing_keeps_tasks_alive(self, make_node_agent: MakeNodeAgent) -> None:
        failing1 = FailingCollector(collect_interval=0.05)
        failing2 = FailingCollector(collect_interval=0.05)
        agent = make_node_agent(
            node_id="test-node-all-fail",
            collectors=[failing1, failing2],
        )
        await agent.start()
        await asyncio.sleep(0.3)

        assert len(agent._collection_loop.tasks) == 2
        assert all(not t.done() for t in agent._collection_loop.tasks)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "miles_ft_" not in response.text

    @pytest.mark.anyio
    async def test_multiple_metrics_exported(self, make_node_agent: MakeNodeAgent) -> None:
        test_collector = TestCollector(
            metrics=[
                GaugeSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=70.0,
                ),
                GaugeSample(
                    name="gpu_memory_used_bytes",
                    labels={"gpu": "0"},
                    value=4096.0,
                ),
            ],
            collect_interval=0.05,
        )
        agent = make_node_agent(
            node_id="test-node-multi",
            collectors=[test_collector],
        )
        await agent.start()
        await asyncio.sleep(0.3)

        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert "gpu_temperature_celsius" in response.text
        assert "gpu_memory_used_bytes" in response.text

    @pytest.mark.anyio
    async def test_per_collector_independent_intervals(self, make_node_agent: MakeNodeAgent) -> None:
        fast_collector = _CountingCollector(collect_interval=0.05)
        slow_collector = _CountingCollector(collect_interval=0.5)
        agent = make_node_agent(
            node_id="test-node-intervals",
            collectors=[fast_collector, slow_collector],
        )
        await agent.start()
        await asyncio.sleep(0.6)

        assert fast_collector.call_count > slow_collector.call_count
        assert fast_collector.call_count >= 5

    @pytest.mark.anyio
    async def test_collect_interval_seconds_overrides_all(self, make_node_agent: MakeNodeAgent) -> None:
        fast_collector = _CountingCollector(collect_interval=0.01)
        slow_collector = _CountingCollector(collect_interval=0.01)
        make_node_agent(
            node_id="test-node-override",
            collectors=[fast_collector, slow_collector],
            collect_interval_seconds=0.05,
        )
        assert fast_collector.collect_interval == 0.05
        assert slow_collector.collect_interval == 0.05


class TestFtNodeAgentLifecycle:
    @pytest.mark.anyio
    async def test_start_twice_is_idempotent(self, make_node_agent: MakeNodeAgent) -> None:
        test_collector = TestCollector(
            metrics=[GaugeSample(name="dummy", labels={}, value=1.0)],
            collect_interval=0.05,
        )
        agent = make_node_agent(
            node_id="test-node-double-start",
            collectors=[test_collector],
        )
        await agent.start()
        first_tasks = list(agent._collection_loop.tasks)
        await agent.start()

        assert agent._collection_loop.tasks == first_tasks

    @pytest.mark.anyio
    async def test_stop_without_start(self) -> None:
        agent = FtNodeAgent(node_id="test-node-no-start")
        await agent.stop()

    @pytest.mark.anyio
    async def test_stop_twice_is_safe(self) -> None:
        agent = FtNodeAgent(
            node_id="test-node-double-stop",
            collectors=[TestCollector(collect_interval=0.05)],
        )
        await agent.start()
        await agent.stop()
        await agent.stop()

    @pytest.mark.anyio
    async def test_start_with_empty_collectors(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-node-empty-collectors", collectors=[])
        await agent.start()
        assert len(agent._collection_loop.tasks) == 0

    @pytest.mark.anyio
    async def test_stop_calls_close_on_all_collectors(self) -> None:
        collector1 = _CountingCollector(collect_interval=0.05)
        collector2 = _CountingCollector(collect_interval=0.05)
        agent = FtNodeAgent(
            node_id="test-node-close",
            collectors=[collector1, collector2],
        )
        await agent.start()
        await asyncio.sleep(0.1)
        await agent.stop()

        assert collector1.closed
        assert collector2.closed

    @pytest.mark.anyio
    async def test_close_failure_does_not_block_other_collectors(self) -> None:
        failing = FailingCloseCollector(collect_interval=0.05)
        good = _CountingCollector(collect_interval=0.05)
        agent = FtNodeAgent(
            node_id="test-node-close-fail",
            collectors=[failing, good],
        )
        await agent.start()
        await asyncio.sleep(0.1)
        await agent.stop()

        assert good.closed


class TestFtNodeAgentDiagnostics:
    @pytest.mark.anyio
    async def test_known_type_dispatches_correctly(self, make_node_agent: MakeNodeAgent) -> None:
        diag = StubDiagnostic(passed=True, details="all good")
        agent = make_node_agent(
            node_id="test-diag-dispatch",
            diagnostics=[diag],
        )
        result = await agent.run_diagnostic("stub")

        assert result.passed is True
        assert result.node_id == "test-diag-dispatch"
        assert result.diagnostic_type == "stub"
        assert result.details == "all good"

    @pytest.mark.anyio
    async def test_unknown_type_raises(self, make_node_agent: MakeNodeAgent) -> None:
        agent = make_node_agent(node_id="test-diag-unknown")
        with pytest.raises(UnknownDiagnosticError, match="unknown diagnostic type"):
            await agent.run_diagnostic("nonexistent")

    @pytest.mark.anyio
    async def test_failing_diagnostic_returns_failure(self, make_node_agent: MakeNodeAgent) -> None:
        diag = StubDiagnostic(passed=False, details="gpu broken", diagnostic_type="failing")
        agent = make_node_agent(
            node_id="test-diag-fail",
            diagnostics=[diag],
        )
        result = await agent.run_diagnostic("failing")

        assert result.passed is False
        assert result.details == "gpu broken"

    @pytest.mark.anyio
    async def test_timeout_returns_failed(self, make_node_agent: MakeNodeAgent) -> None:
        diag = SlowDiagnostic(sleep_seconds=300.0)
        agent = make_node_agent(
            node_id="test-diag-timeout",
            diagnostics=[diag],
        )
        result = await agent.run_diagnostic("slow", timeout_seconds=0)

        assert result.passed is False
        assert "timed out" in result.details

    @pytest.mark.anyio
    async def test_multiple_diagnostics_registered(self, make_node_agent: MakeNodeAgent) -> None:
        stub = StubDiagnostic(passed=True)
        failing = StubDiagnostic(passed=False, details="diagnostic failed", diagnostic_type="failing")
        agent = make_node_agent(
            node_id="test-diag-multi",
            diagnostics=[stub, failing],
        )
        r1 = await agent.run_diagnostic("stub")
        r2 = await agent.run_diagnostic("failing")

        assert r1.passed is True
        assert r2.passed is False


class TestFtNodeAgentKwargsPassthrough:
    @pytest.mark.anyio
    async def test_kwargs_forwarded_to_diagnostic_run(self, make_node_agent: MakeNodeAgent) -> None:
        """run_diagnostic passes **kwargs through to diagnostic.run()."""
        received_kwargs: dict[str, object] = {}

        class _CapturingDiagnostic(StubDiagnostic):
            async def run(
                self,
                node_id: str,
                timeout_seconds: int = 120,
                **kwargs: object,
            ) -> DiagnosticResult:
                received_kwargs.update(kwargs)
                return await super().run(node_id=node_id, timeout_seconds=timeout_seconds)

        agent = make_node_agent(
            node_id="test-kwargs",
            diagnostics=[_CapturingDiagnostic(passed=True)],
        )
        await agent.run_diagnostic("stub", master_addr="10.0.0.1", master_port=29500)

        assert received_kwargs["master_addr"] == "10.0.0.1"
        assert received_kwargs["master_port"] == 29500


class TestFtNodeAgentDiagnosticException:
    @pytest.mark.anyio
    async def test_diagnostic_exception_returns_failed(self, make_node_agent: MakeNodeAgent) -> None:
        """Diagnostic that raises (not timeout) should return passed=False."""

        class _ExplodingDiagnostic(StubDiagnostic):
            async def run(self, node_id: str, timeout_seconds: int = 120) -> None:
                raise RuntimeError("diagnostic exploded")

        agent = make_node_agent(
            node_id="test-diag-explode",
            diagnostics=[_ExplodingDiagnostic(passed=True)],
        )
        result = await agent.run_diagnostic("stub")
        assert result.passed is False
        assert "exception" in result.details
