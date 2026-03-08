import pytest

from miles.utils.ft.adapters.stubs import StubNodeManager, StubNotifier, StubTrainingJob
from miles.utils.ft.controller.types import DiagnosticOrchestratorProtocol
from miles.utils.ft.adapters.types import (
    JobStatus,
    NodeManagerProtocol,
    NotifierProtocol,
    TrainingJobProtocol,
)


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
    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def mark_node_bad(self, node_id: str, reason: str) -> None:
                pass

            async def unmark_node_bad(self, node_id: str) -> None:
                pass

            async def get_bad_nodes(self) -> list[str]:
                return []

        assert isinstance(_Conforming(), NodeManagerProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _MissingGetBadNodes:
            async def mark_node_bad(self, node_id: str, reason: str) -> None:
                pass

            async def unmark_node_bad(self, node_id: str) -> None:
                pass

        assert not isinstance(_MissingGetBadNodes(), NodeManagerProtocol)

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl:
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


class TestTrainingJobProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def stop_training(self, timeout_seconds: int = 300) -> None:
                pass

            async def submit_training(self) -> str:
                return "run-123"

            async def get_training_status(self) -> JobStatus:
                return JobStatus.RUNNING

        assert isinstance(_Conforming(), TrainingJobProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _MissingSubmit:
            async def stop_training(self, timeout_seconds: int = 300) -> None:
                pass

            async def get_training_status(self) -> JobStatus:
                return JobStatus.RUNNING

        assert not isinstance(_MissingSubmit(), TrainingJobProtocol)

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl:
            def __init__(self) -> None:
                self._status = JobStatus.PENDING

            async def stop_training(self, timeout_seconds: int = 300) -> None:
                self._status = JobStatus.STOPPED

            async def submit_training(self) -> str:
                self._status = JobStatus.RUNNING
                return "run-abc"

            async def get_training_status(self) -> JobStatus:
                return self._status

        instance: TrainingJobProtocol = _Impl()
        run_id = await instance.submit_training()
        assert run_id == "run-abc"
        assert await instance.get_training_status() == JobStatus.RUNNING
        await instance.stop_training(timeout_seconds=10)
        assert await instance.get_training_status() == JobStatus.STOPPED


class TestNotifierProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def send(self, title: str, content: str, severity: str) -> None:
                pass

            async def aclose(self) -> None:
                pass

        assert isinstance(_Conforming(), NotifierProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _MissingSend:
            async def aclose(self) -> None:
                pass

        assert not isinstance(_MissingSend(), NotifierProtocol)

    @pytest.mark.anyio
    async def test_methods_callable_with_expected_signatures(self) -> None:
        class _Impl:
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

    def test_stub_training_job_satisfies_protocol(self) -> None:
        assert isinstance(StubTrainingJob(), TrainingJobProtocol)

    def test_stub_notifier_satisfies_protocol(self) -> None:
        assert isinstance(StubNotifier(), NotifierProtocol)


class TestDiagnosticOrchestratorProtocol:
    def test_conforming_class_passes_isinstance(self) -> None:
        class _Conforming:
            async def run_diagnostic_pipeline(
                self,
                pre_executors: object = None,
            ) -> object:
                return None

        assert isinstance(_Conforming(), DiagnosticOrchestratorProtocol)

    def test_missing_method_fails_isinstance(self) -> None:
        class _Empty:
            pass

        assert not isinstance(_Empty(), DiagnosticOrchestratorProtocol)
