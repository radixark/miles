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
    async def test_mark_node_bad_does_not_raise(self) -> None:
        manager = StubNodeManager()
        await manager.mark_node_bad(node_id="node-1", reason="gpu failure")

class TestStubMainJob:
    async def test_submit_returns_unique_run_id(self) -> None:
        job = StubMainJob()
        run_id_1 = await job.start()
        run_id_2 = await job.start()
        assert isinstance(run_id_1, str)
        assert len(run_id_1) == 8
        assert run_id_1 != run_id_2

    async def test_get_job_status_returns_running(self) -> None:
        job = StubMainJob()
        status = await job.get_status()
        assert status == JobStatus.RUNNING

    async def test_stop_job_does_not_raise(self) -> None:
        job = StubMainJob()
        await job.stop(timeout_seconds=10)
