"""Tests for conftest.py builder helpers."""

import pytest

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import DiagnosticResult, MetricSample
from miles.utils.ft.platform.protocols import JobStatus
from tests.fast.utils.ft.conftest import (
    FakeNodeAgent,
    FakeNodeManager,
    FakeTrainingJob,
    TestCollector,
    make_fake_metric_store,
    make_fake_mini_wandb,
    make_metric,
)


class TestMakeMetric:
    def test_basic(self) -> None:
        m = make_metric("gpu_temp", 75.0)
        assert m.name == "gpu_temp"
        assert m.value == 75.0
        assert m.labels == {}

    def test_with_labels(self) -> None:
        m = make_metric("gpu_temp", 75.0, labels={"gpu": "0"})
        assert m.labels == {"gpu": "0"}


class TestMakeFakeMetricStore:
    def test_empty_store(self) -> None:
        store = make_fake_metric_store()
        df = store.query_latest("anything")
        assert df.is_empty()

    def test_with_metrics(self) -> None:
        metrics = [
            make_metric("gpu_temp", 75.0, labels={"gpu": "0"}),
            make_metric("gpu_temp", 82.0, labels={"gpu": "1"}),
        ]
        store = make_fake_metric_store(metrics=metrics)
        df = store.query_latest("gpu_temp")
        assert len(df) == 2


class TestMakeFakeMiniWandb:
    def test_empty_wandb(self) -> None:
        wandb = make_fake_mini_wandb()
        assert wandb.latest(metric_name="loss", rank=0) is None

    def test_with_steps(self) -> None:
        wandb = make_fake_mini_wandb(steps={
            1: {"loss": 3.0, "grad_norm": 1.0},
            2: {"loss": 2.5, "grad_norm": 0.8},
        })
        assert wandb.latest(metric_name="loss", rank=0) == 2.5
        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        assert len(result) == 2
        assert result[0] == (1, 3.0)
        assert result[1] == (2, 2.5)


class TestFakeNodeManager:
    @pytest.mark.asyncio
    async def test_mark_and_get_bad_nodes(self) -> None:
        manager = FakeNodeManager()
        await manager.mark_node_bad(node_id="node-1", reason="gpu failure")
        await manager.mark_node_bad(node_id="node-2", reason="network error")
        assert await manager.get_bad_nodes() == ["node-1", "node-2"]

    @pytest.mark.asyncio
    async def test_unmark_node(self) -> None:
        manager = FakeNodeManager()
        await manager.mark_node_bad(node_id="node-1", reason="test")
        await manager.unmark_node_bad(node_id="node-1")
        assert await manager.get_bad_nodes() == []

    @pytest.mark.asyncio
    async def test_is_node_bad(self) -> None:
        manager = FakeNodeManager()
        assert not manager.is_node_bad("node-1")
        await manager.mark_node_bad(node_id="node-1", reason="test")
        assert manager.is_node_bad("node-1")

    @pytest.mark.asyncio
    async def test_unmark_nonexistent_node(self) -> None:
        manager = FakeNodeManager()
        await manager.unmark_node_bad(node_id="node-1")


class TestFakeTrainingJob:
    @pytest.mark.asyncio
    async def test_status_sequence(self) -> None:
        job = FakeTrainingJob(status_sequence=[
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.FAILED,
        ])
        assert await job.get_training_status() == JobStatus.PENDING
        assert await job.get_training_status() == JobStatus.RUNNING
        assert await job.get_training_status() == JobStatus.FAILED
        assert await job.get_training_status() == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_default_status_is_running(self) -> None:
        job = FakeTrainingJob()
        assert await job.get_training_status() == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_sets_flag(self) -> None:
        job = FakeTrainingJob()
        assert not job._stopped
        await job.stop_training()
        assert job._stopped

    @pytest.mark.asyncio
    async def test_submit_resets_call_count(self) -> None:
        job = FakeTrainingJob(status_sequence=[
            JobStatus.PENDING,
            JobStatus.RUNNING,
        ])
        assert await job.get_training_status() == JobStatus.PENDING
        await job.submit_training()
        assert job._submitted
        assert await job.get_training_status() == JobStatus.PENDING


class TestTestCollector:
    def test_is_base_collector(self) -> None:
        assert issubclass(TestCollector, BaseCollector)

    @pytest.mark.asyncio()
    async def test_default_empty_metrics(self) -> None:
        collector = TestCollector()
        output = await collector.collect()
        assert output.metrics == []

    @pytest.mark.asyncio()
    async def test_with_preset_metrics(self) -> None:
        metrics = [MetricSample(name="temp", labels={"gpu": "0"}, value=75.0)]
        collector = TestCollector(metrics=metrics)
        output = await collector.collect()
        assert output.metrics == metrics

    @pytest.mark.asyncio()
    async def test_set_metrics(self) -> None:
        collector = TestCollector()
        new_metrics = [MetricSample(name="temp", labels={}, value=80.0)]
        collector.set_metrics(new_metrics)
        output = await collector.collect()
        assert output.metrics == new_metrics


class TestFakeNodeAgent:
    @pytest.mark.asyncio()
    async def test_run_diagnostic(self) -> None:
        result = DiagnosticResult(
            diagnostic_type="gpu_check",
            node_id="node-0",
            passed=True,
            details="all good",
        )
        agent = FakeNodeAgent(diagnostic_results={"gpu_check": result})
        assert await agent.run_diagnostic("gpu_check") == result

    @pytest.mark.asyncio()
    async def test_cleanup_training_processes(self) -> None:
        agent = FakeNodeAgent()
        assert agent.cleanup_called is False

        await agent.cleanup_training_processes("job-42")
        assert agent.cleanup_called is True
        assert agent.cleanup_job_id == "job-42"

    def test_default_construction(self) -> None:
        agent = FakeNodeAgent()
        assert agent.cleanup_called is False
        assert agent.cleanup_job_id is None
