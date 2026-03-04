import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.collectors.stub import StubCollector
from miles.utils.ft.agents.node_agent import FtNodeAgent
from miles.utils.ft.models import CollectorOutput, MetricSample
from tests.fast.utils.ft.conftest import TestCollector


class _FailingCollector(BaseCollector):
    async def collect(self) -> CollectorOutput:
        raise RuntimeError("simulated collector failure")


class _CountingCollector(BaseCollector):
    def __init__(self, collect_interval: float = 10.0) -> None:
        self.collect_interval = collect_interval
        self.call_count = 0
        self.closed = False

    async def collect(self) -> CollectorOutput:
        self.call_count += 1
        return CollectorOutput(metrics=[
            MetricSample(name="count", labels={}, value=float(self.call_count)),
        ])

    async def close(self) -> None:
        self.closed = True


class TestFtNodeAgentExporter:
    @pytest_asyncio.fixture()
    async def agent(self) -> AsyncIterator[FtNodeAgent]:
        agent = FtNodeAgent(node_id="test-node-0")
        yield agent
        await agent.stop()

    @pytest.mark.asyncio()
    async def test_exporter_returns_prometheus_format(
        self, agent: FtNodeAgent
    ) -> None:
        address = agent.get_exporter_address()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{address}/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    @pytest.mark.asyncio()
    async def test_exporter_address_has_port(self, agent: FtNodeAgent) -> None:
        address = agent.get_exporter_address()
        assert address.startswith("http://localhost:")
        port = int(address.split(":")[-1])
        assert port > 0

    @pytest.mark.asyncio()
    async def test_stub_collector_no_custom_metrics(self) -> None:
        agent = FtNodeAgent(
            node_id="test-node-stub",
            collectors=[StubCollector()],
        )
        try:
            agent._update_exporter([])

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert response.status_code == 200
            assert "miles_ft_node_" not in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_update_exporter_creates_gauges(self) -> None:
        agent = FtNodeAgent(node_id="test-node-gauge")
        try:
            metrics = [
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=75.0,
                ),
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "1"},
                    value=80.0,
                ),
            ]
            agent._update_exporter(metrics)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            text = response.text
            assert 'miles_ft_node_gpu_temperature_celsius{gpu="0"}' in text
            assert 'miles_ft_node_gpu_temperature_celsius{gpu="1"}' in text
            assert "75.0" in text
            assert "80.0" in text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_update_exporter_unlabeled_metric(self) -> None:
        agent = FtNodeAgent(node_id="test-node-unlabeled")
        try:
            metrics = [
                MetricSample(name="uptime_seconds", labels={}, value=123.0),
            ]
            agent._update_exporter(metrics)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert "miles_ft_node_uptime_seconds" in response.text
            assert "123.0" in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_update_exporter_overwrites_value(self) -> None:
        agent = FtNodeAgent(node_id="test-node-overwrite")
        try:
            agent._update_exporter([
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=60.0,
                ),
            ])
            agent._update_exporter([
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=90.0,
                ),
            ])

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert "90.0" in response.text
        finally:
            await agent.stop()


class TestFtNodeAgentCollectionLoop:
    @pytest.mark.asyncio()
    async def test_collection_loop_updates_exporter(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=65.0,
                ),
            ],
            collect_interval=0.05,
        )
        agent = FtNodeAgent(
            node_id="test-node-loop",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert 'miles_ft_node_gpu_temperature_celsius{gpu="0"}' in response.text
            assert "65.0" in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_stop_cancels_tasks(self) -> None:
        test_collector = TestCollector(
            metrics=[MetricSample(name="dummy", labels={}, value=1.0)],
            collect_interval=0.05,
        )
        agent = FtNodeAgent(
            node_id="test-node-stop",
            collectors=[test_collector],
        )
        await agent.start()

        assert len(agent._collector_tasks) == 1
        await agent.stop()
        assert len(agent._collector_tasks) == 0

    @pytest.mark.asyncio()
    async def test_failing_collector_does_not_crash_loop(self) -> None:
        good_collector = TestCollector(
            metrics=[MetricSample(name="good_metric", labels={}, value=42.0)],
            collect_interval=0.05,
        )
        failing = _FailingCollector()
        failing.collect_interval = 0.05
        agent = FtNodeAgent(
            node_id="test-node-fail",
            collectors=[failing, good_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert "miles_ft_node_good_metric" in response.text
            assert "42.0" in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_all_collectors_failing_keeps_tasks_alive(self) -> None:
        failing1 = _FailingCollector()
        failing1.collect_interval = 0.05
        failing2 = _FailingCollector()
        failing2.collect_interval = 0.05
        agent = FtNodeAgent(
            node_id="test-node-all-fail",
            collectors=[failing1, failing2],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            assert len(agent._collector_tasks) == 2
            assert all(not t.done() for t in agent._collector_tasks)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert response.status_code == 200
            assert "miles_ft_node_" not in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_multiple_metrics_exported(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=70.0,
                ),
                MetricSample(
                    name="gpu_memory_used_bytes",
                    labels={"gpu": "0"},
                    value=4096.0,
                ),
            ],
            collect_interval=0.05,
        )
        agent = FtNodeAgent(
            node_id="test-node-multi",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            address = agent.get_exporter_address()
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{address}/metrics")

            assert "miles_ft_node_gpu_temperature_celsius" in response.text
            assert "miles_ft_node_gpu_memory_used_bytes" in response.text
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_per_collector_independent_intervals(self) -> None:
        fast_collector = _CountingCollector(collect_interval=0.05)
        slow_collector = _CountingCollector(collect_interval=0.5)
        agent = FtNodeAgent(
            node_id="test-node-intervals",
            collectors=[fast_collector, slow_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.6)

            assert fast_collector.call_count > slow_collector.call_count
            assert fast_collector.call_count >= 5
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_collect_interval_seconds_overrides_all(self) -> None:
        fast_collector = _CountingCollector(collect_interval=0.01)
        slow_collector = _CountingCollector(collect_interval=0.01)
        agent = FtNodeAgent(
            node_id="test-node-override",
            collectors=[fast_collector, slow_collector],
            collect_interval_seconds=0.05,
        )
        try:
            assert fast_collector.collect_interval == 0.05
            assert slow_collector.collect_interval == 0.05
        finally:
            await agent.stop()


class TestFtNodeAgentLifecycle:
    @pytest.mark.asyncio()
    async def test_start_twice_is_idempotent(self) -> None:
        test_collector = TestCollector(
            metrics=[MetricSample(name="dummy", labels={}, value=1.0)],
            collect_interval=0.05,
        )
        agent = FtNodeAgent(
            node_id="test-node-double-start",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            first_tasks = list(agent._collector_tasks)
            await agent.start()

            assert agent._collector_tasks == first_tasks
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_stop_without_start(self) -> None:
        agent = FtNodeAgent(node_id="test-node-no-start")
        await agent.stop()

    @pytest.mark.asyncio()
    async def test_stop_twice_is_safe(self) -> None:
        agent = FtNodeAgent(
            node_id="test-node-double-stop",
            collectors=[TestCollector(collect_interval=0.05)],
        )
        await agent.start()
        await agent.stop()
        await agent.stop()

    @pytest.mark.asyncio()
    async def test_start_with_empty_collectors(self) -> None:
        agent = FtNodeAgent(node_id="test-node-empty-collectors", collectors=[])
        try:
            await agent.start()
            assert len(agent._collector_tasks) == 0
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
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

    @pytest.mark.asyncio()
    async def test_close_failure_does_not_block_other_collectors(self) -> None:
        class _FailingCloseCollector(BaseCollector):
            def __init__(self) -> None:
                self.collect_interval = 0.05

            async def collect(self) -> CollectorOutput:
                return CollectorOutput(metrics=[])

            async def close(self) -> None:
                raise RuntimeError("close failed")

        failing = _FailingCloseCollector()
        good = _CountingCollector(collect_interval=0.05)
        agent = FtNodeAgent(
            node_id="test-node-close-fail",
            collectors=[failing, good],
        )
        await agent.start()
        await asyncio.sleep(0.1)
        await agent.stop()

        assert good.closed


class TestFtNodeAgentStubMethods:
    @pytest_asyncio.fixture()
    async def agent(self) -> AsyncIterator[FtNodeAgent]:
        agent = FtNodeAgent(node_id="test-node-stubs")
        yield agent
        await agent.stop()

    @pytest.mark.asyncio()
    async def test_collect_logs_raises(self, agent: FtNodeAgent) -> None:
        with pytest.raises(NotImplementedError, match="collect_logs"):
            await agent.collect_logs()

    @pytest.mark.asyncio()
    async def test_run_diagnostic_raises(self, agent: FtNodeAgent) -> None:
        with pytest.raises(NotImplementedError, match="run_diagnostic"):
            await agent.run_diagnostic("gpu_check")

    @pytest.mark.asyncio()
    async def test_cleanup_training_processes_raises(
        self, agent: FtNodeAgent
    ) -> None:
        with pytest.raises(NotImplementedError, match="cleanup_training_processes"):
            await agent.cleanup_training_processes("job-123")
