import asyncio
import subprocess
import threading
from pathlib import Path

import pytest
from tests.fast.utils.soak.soak_fakes import _RecordingProcesses, _RecordingReleaseRemoval
from tests.utils.soak.core import teardown as soak_teardown
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakTeardownEvent
from tests.utils.soak.core.teardown import teardown_run

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME

_RUN_ID = "260926-120000-000"


def _ray_config(submission_id: str | None = "miles-soak-abc") -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id=_RUN_ID, ray_submission_id=submission_id)


def _kubernetes_config(namespace: str = "rl") -> ExecuteTrainConfig:
    return ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace=namespace,
        run_id=_RUN_ID,
        deploy_component=DeployComponent.TRAINER,
        deploy_instance_id="b",
    )


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> _RecordingProcesses:
    fake = _RecordingProcesses()
    monkeypatch.setattr(soak_teardown, "run_process", fake)
    return fake


@pytest.fixture
def removal(monkeypatch: pytest.MonkeyPatch) -> _RecordingReleaseRemoval:
    fake = _RecordingReleaseRemoval()
    monkeypatch.setattr(soak_teardown, "remove_release_and_wait", fake)
    return fake


class TestRayTeardown:
    async def test_only_the_owned_submission_is_stopped_and_its_output_kept(
        self,
        tmp_path: Path,
        processes: _RecordingProcesses,
        removal: _RecordingReleaseRemoval,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Ray soak stops exactly its own job on the local dashboard and archives the stop output."""
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
        log = EventLog(tmp_path / "events.jsonl")

        await teardown_run(config=_ray_config(), event_log=log, evidence_dir=tmp_path / "evidence")

        assert processes.calls == [["ray", "job", "stop", "--address", "http://127.0.0.1:8265", "miles-soak-abc"]]
        assert removal.calls == []
        assert (tmp_path / "evidence" / "ray-job-stop.log").read_text() == "stopped\n"
        assert log.events == [
            SoakTeardownEvent(timestamp=log.events[0].timestamp, resource="ray-job:miles-soak-abc", returned=True)
        ]

    async def test_a_configured_ray_address_replaces_the_local_dashboard(
        self, tmp_path: Path, processes: _RecordingProcesses, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With RAY_ADDRESS set the stop command lets Ray pick the configured cluster."""
        monkeypatch.setenv("RAY_ADDRESS", "http://ray:8265")

        await teardown_run(config=_ray_config(), event_log=EventLog(tmp_path / "e.jsonl"), evidence_dir=tmp_path)

        assert processes.calls == [["ray", "job", "stop", "miles-soak-abc"]]

    @pytest.mark.parametrize("submission_id", [None, ""])
    async def test_a_ray_run_without_an_owned_submission_is_refused(
        self, tmp_path: Path, processes: _RecordingProcesses, submission_id: str | None
    ) -> None:
        """Without its own submission id cleanup could stop someone else's job, so it refuses."""
        log = EventLog(tmp_path / "events.jsonl")

        with pytest.raises(AssertionError, match="owned Ray submission"):
            await teardown_run(config=_ray_config(submission_id), event_log=log, evidence_dir=tmp_path)
        assert processes.calls == []

    async def test_a_failed_stop_is_recorded_and_reraised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing ray job stop leaves a not-returned teardown event and fails the soak."""
        error = subprocess.CalledProcessError(1, ["ray"])
        monkeypatch.setattr(soak_teardown, "run_process", _RecordingProcesses(error=error))
        log = EventLog(tmp_path / "events.jsonl")

        with pytest.raises(subprocess.CalledProcessError):
            await teardown_run(config=_ray_config(), event_log=log, evidence_dir=tmp_path)

        [event] = log.events
        assert isinstance(event, SoakTeardownEvent)
        assert (event.resource, event.returned) == ("ray-job:miles-soak-abc", False)
        assert "CalledProcessError" in event.error


class TestKubernetesTeardown:
    async def test_only_the_release_of_this_run_is_removed(
        self, tmp_path: Path, processes: _RecordingProcesses, removal: _RecordingReleaseRemoval
    ) -> None:
        """A Kubernetes soak uninstalls the release named by its own run, component and instance."""
        log = EventLog(tmp_path / "events.jsonl")

        await teardown_run(config=_kubernetes_config(), event_log=log, evidence_dir=tmp_path)

        release = f"{CHART_NAME}-{_RUN_ID}-trainer-b"
        assert removal.calls == [(release, "rl")]
        assert processes.calls == []
        [event] = log.events
        assert isinstance(event, SoakTeardownEvent)
        assert (event.resource, event.returned, event.error) == (f"helm:rl/{release}", True, None)

    async def test_a_kubernetes_run_without_a_namespace_is_refused(
        self, tmp_path: Path, removal: _RecordingReleaseRemoval
    ) -> None:
        """An empty namespace could resolve to another namespace's release, so cleanup refuses."""
        log = EventLog(tmp_path / "events.jsonl")

        with pytest.raises(AssertionError, match="explicit namespace"):
            await teardown_run(config=_kubernetes_config(namespace=""), event_log=log, evidence_dir=tmp_path)
        assert removal.calls == []
        assert log.events == []

    async def test_a_removal_timeout_is_recorded_and_reraised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A removal that never finishes is bounded and leaves a failed teardown event."""
        release_removal = threading.Event()
        monkeypatch.setattr(
            soak_teardown, "remove_release_and_wait", _RecordingReleaseRemoval(block=release_removal)
        )
        monkeypatch.setattr(soak_teardown, "_TEARDOWN_TIMEOUT_SECONDS", 0.05)
        log = EventLog(tmp_path / "events.jsonl")

        try:
            with pytest.raises(TimeoutError):
                await teardown_run(config=_kubernetes_config(), event_log=log, evidence_dir=tmp_path)
        finally:
            release_removal.set()

        [event] = log.events
        assert isinstance(event, SoakTeardownEvent)
        assert event.returned is False
        assert "TimeoutError" in event.error

    async def test_a_cancelled_teardown_is_recorded_and_the_cancellation_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling cleanup still leaves evidence that the resource may remain."""
        release_removal = threading.Event()
        fake = _RecordingReleaseRemoval(block=release_removal)
        monkeypatch.setattr(soak_teardown, "remove_release_and_wait", fake)
        log = EventLog(tmp_path / "events.jsonl")

        task = asyncio.create_task(teardown_run(config=_kubernetes_config(), event_log=log, evidence_dir=tmp_path))
        try:
            async with asyncio.timeout(5):
                while not fake.calls:
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release_removal.set()

        [event] = log.events
        assert isinstance(event, SoakTeardownEvent)
        assert event.returned is False
        assert "CancelledError" in event.error
