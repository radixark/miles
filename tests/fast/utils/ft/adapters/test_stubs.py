from miles.utils.ft.adapters.stubs import NullMetadataProvider, StubMainJob, StubNodeManager
from miles.utils.ft.adapters.types import JobStatus


class TestNullMetadataProvider:
    def test_returns_empty_dict(self) -> None:
        provider = NullMetadataProvider()
        assert provider.get_metadata() == {}

    def test_returns_new_dict_each_call(self) -> None:
        provider = NullMetadataProvider()
        first = provider.get_metadata()
        second = provider.get_metadata()
        assert first == second == {}
        assert first is not second


class TestStubNodeManager:
    async def test_mark_and_get_bad_nodes(self) -> None:
        manager = StubNodeManager()
        await manager.mark_node_bad(node_id="node-1", reason="gpu failure")
        result = await manager.get_bad_nodes()
        assert result == []

    async def test_unmark_node_bad(self) -> None:
        manager = StubNodeManager()
        await manager.unmark_node_bad(node_id="node-1")


class TestStubMainJob:
    async def test_submit_returns_unique_run_id(self) -> None:
        job = StubMainJob()
        run_id_1 = await job.submit_job()
        run_id_2 = await job.submit_job()
        assert isinstance(run_id_1, str)
        assert len(run_id_1) == 8
        assert run_id_1 != run_id_2

    async def test_get_job_status_returns_running(self) -> None:
        job = StubMainJob()
        status = await job.get_job_status()
        assert status == JobStatus.RUNNING

    async def test_stop_job_does_not_raise(self) -> None:
        job = StubMainJob()
        await job.stop_job(timeout_seconds=10)
