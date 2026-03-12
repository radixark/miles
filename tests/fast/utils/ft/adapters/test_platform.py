import pytest

from miles.utils.ft.adapters.stubs import StubMainJob, StubNodeManager, StubNotifier
from miles.utils.ft.adapters.types import JobStatus, MainJobProtocol, NodeManagerProtocol, NotifierProtocol
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol


class TestJobStatus:
    def test_enum_values(self) -> None:
        assert JobStatus.RUNNING == "running"
        assert JobStatus.STOPPED == "stopped"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.PENDING == "pending"

    def test_all_members(self) -> None:
        assert set(JobStatus) == {
            JobStatus.RUNNING,
            JobStatus.STOPPED,
            JobStatus.FAILED,
            JobStatus.PENDING,
        }

    def test_string_conversion(self) -> None:
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus("failed") == JobStatus.FAILED


class TestNodeManagerProtocol:
    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _MissingGetBadNodes(NodeManagerProtocol):
            async def mark_node_bad(self, node_id: str, reason: str) -> None:
                pass

            async def unmark_node_bad(self, node_id: str) -> None:
                pass

        with pytest.raises(TypeError):
            _MissingGetBadNodes()

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl(NodeManagerProtocol):
            def __init__(self) -> None:
                self._bad: dict[str, str] = {}

            async def mark_node_bad(self, node_id: str, reason: str) -> None:
                self._bad[node_id] = reason

            async def unmark_node_bad(self, node_id: str) -> None:
                self._bad.pop(node_id, None)

            async def get_bad_nodes(self) -> list[str]:
                return list(self._bad.keys())

        instance: NodeManagerProtocol = _Impl()
        await instance.mark_node_bad(node_id="n-0", reason="bad gpu")
        assert await instance.get_bad_nodes() == ["n-0"]
        await instance.unmark_node_bad(node_id="n-0")
        assert await instance.get_bad_nodes() == []


class TestMainJobProtocol:
    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _MissingSubmit(MainJobProtocol):
            async def stop(self, timeout_seconds: int = 300) -> None:
                pass

            async def get_status(self) -> JobStatus:
                return JobStatus.RUNNING

        with pytest.raises(TypeError):
            _MissingSubmit()

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl(MainJobProtocol):
            def __init__(self) -> None:
                self._status = JobStatus.PENDING

            async def stop(self, timeout_seconds: int = 300) -> None:
                self._status = JobStatus.STOPPED

            async def start(self) -> str:
                self._status = JobStatus.RUNNING
                return "run-abc"

            async def get_status(self) -> JobStatus:
                return self._status

        instance: MainJobProtocol = _Impl()
        run_id = await instance.start()
        assert run_id == "run-abc"
        assert await instance.get_status() == JobStatus.RUNNING
        await instance.stop(timeout_seconds=10)
        assert await instance.get_status() == JobStatus.STOPPED


class TestNotifierProtocol:
    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _MissingSend(NotifierProtocol):
            async def aclose(self) -> None:
                pass

        with pytest.raises(TypeError):
            _MissingSend()

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl(NotifierProtocol):
            def __init__(self) -> None:
                self.sent: list[tuple[str, str, str]] = []
                self.closed = False

            async def send(self, title: str, content: str, severity: str) -> None:
                self.sent.append((title, content, severity))

            async def aclose(self) -> None:
                self.closed = True

        instance: NotifierProtocol = _Impl()
        await instance.send(title="alert", content="gpu down", severity="critical")
        assert instance.sent == [("alert", "gpu down", "critical")]
        await instance.aclose()
        assert instance.closed


class TestStubProtocolCompliance:
    def test_stub_node_manager_satisfies_protocol(self) -> None:
        assert isinstance(StubNodeManager(), NodeManagerProtocol)

    def test_stub_main_job_satisfies_protocol(self) -> None:
        assert isinstance(StubMainJob(), MainJobProtocol)

    def test_stub_notifier_satisfies_protocol(self) -> None:
        assert isinstance(StubNotifier(), NotifierProtocol)


class TestDiagnosticOrchestratorProtocol:
    def test_incomplete_subclass_raises_type_error(self) -> None:
        class _Empty(DiagnosticOrchestratorProtocol):
            pass

        with pytest.raises(TypeError):
            _Empty()
