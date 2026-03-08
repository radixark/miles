from miles.utils.ft.adapters.stubs import StubNodeManager, StubTrainingJob
from miles.utils.ft.adapters.types import JobStatus


class TestStubNodeManager:
    async def test_mark_and_get_bad_nodes(self) -> None:
        manager = StubNodeManager()
        await manager.mark_node_bad(node_id="node-1", reason="gpu failure")
        result = await manager.get_bad_nodes()
        assert result == []

    async def test_unmark_node_bad(self) -> None:
        manager = StubNodeManager()
        await manager.unmark_node_bad(node_id="node-1")


class TestStubTrainingJob:
    async def test_submit_returns_unique_run_id(self) -> None:
        job = StubTrainingJob()
        run_id_1 = await job.submit_training()
        run_id_2 = await job.submit_training()
        assert isinstance(run_id_1, str)
        assert len(run_id_1) == 8
        assert run_id_1 != run_id_2

    async def test_get_training_status_returns_running(self) -> None:
        job = StubTrainingJob()
        status = await job.get_training_status()
        assert status == JobStatus.RUNNING

    async def test_stop_training_does_not_raise(self) -> None:
        job = StubTrainingJob()
        await job.stop_training(timeout_seconds=10)
