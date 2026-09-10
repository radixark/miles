import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from tests.utils.soak import teardown
from tests.utils.soak.state import EventLog, SoakTeardownEvent
from tests.utils.soak.utils import create_soak_config

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


def test_each_ray_soak_owns_a_distinct_submission_without_mutating_the_input() -> None:
    """Repeated soaks cannot accidentally share a remote job cleanup identity."""
    original = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id="shared")
    first, second = create_soak_config(original), create_soak_config(original)
    assert first.ray_submission_id != second.ray_submission_id
    assert first.ray_submission_id and second.ray_submission_id
    assert original.ray_submission_id is None
    assert first.run_id == second.run_id == original.run_id


@pytest.mark.parametrize("failed", [False, True])
async def test_ray_teardown_targets_only_the_owned_submission_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    """A failed remote stop is preserved instead of being reported as completed cleanup."""
    config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, ray_submission_id="owned-job")
    log = EventLog()
    calls: list[list[str]] = []

    async def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if failed:
            raise subprocess.CalledProcessError(returncode=1, cmd=args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="stopped", stderr="")

    monkeypatch.setattr(teardown, "run_command", command)
    if failed:
        with pytest.raises(subprocess.CalledProcessError):
            await teardown._teardown_run(config=config, event_log=log, evidence_dir=tmp_path)
    else:
        await teardown._teardown_run(config=config, event_log=log, evidence_dir=tmp_path)
    assert len(calls) == 1 and calls[0][:3] == ["ray", "job", "stop"] and calls[0][-1] == "owned-job"
    assert "--no-wait" not in calls[0]
    assert isinstance(log.events[-1], SoakTeardownEvent)
    assert log.events[-1].resource == "ray-job:owned-job"
    assert log.events[-1].returned is not failed


async def test_release_teardown_waits_for_pods_even_after_helm_has_disappeared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing Helm release is insufficient while its pods still occupy resources."""
    config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="owned-ns", run_id="owned")
    log = EventLog()
    calls: list[list[str]] = []
    pod_reads = 0

    async def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal pod_reads
        calls.append(args)
        assert args[args.index("--namespace") + 1] == "owned-ns"
        if args[:3] == ["kubectl", "get", "pods"]:
            pod_reads += 1
            output = {"items": [{"metadata": {"name": "remaining"}}] if pod_reads == 1 else []}
        else:
            output = []
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(teardown, "run_command", command)
    monkeypatch.setattr(teardown.asyncio, "sleep", AsyncMock())
    await teardown._teardown_run(config=config, event_log=log, evidence_dir=tmp_path)
    assert pod_reads == 2
    release = calls[0][2]
    for args in calls[1:]:
        assert release in " ".join(args)
    assert isinstance(log.events[-1], SoakTeardownEvent) and log.events[-1].returned
