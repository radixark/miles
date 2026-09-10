import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import progress, soak_observer
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import compute_hot_restart_workloads

from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER


@pytest.mark.parametrize("failure", ["missing_file", "timeout"])
async def test_progress_read_failures_withhold_targets_and_recover_on_the_next_poll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    """Rolling event files aside or a stalled disk cannot terminate the hot restart observer."""
    observer = soak_observer.HotRestartSoakObserver(
        base_url="http://control",
        cell_types=set(),
        namespace="ns",
        release="release",
        trainer_id="actor",
        checkpoint_dir=tmp_path / "ckpt",
        events_dir=tmp_path / "events",
    )
    progress_reads = 0
    workloads = [
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": name, "uid": name, "generation": 1},
            "spec": {"template": {"spec": {"containers": []}}},
        }
        for name in sorted(compute_hot_restart_workloads("release"))
    ]

    async def command(argv: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        nonlocal progress_reads
        assert 0 < timeout_seconds <= 60
        if argv[0] == sys.executable:
            progress_reads += 1
            if progress_reads == 1:
                if failure == "timeout":
                    raise TimeoutError("slow shared disk")
                raise subprocess.CalledProcessError(returncode=1, cmd=argv, stderr="FileNotFoundError")
            payload = {"last_saved_iteration": 3, "last_finished_rollout_id": 4}
        elif argv[2] == "job":
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")
        elif argv[2] == "pods":
            payload = {"items": [{"metadata": {"name": "pod", "uid": "pod-uid"}}]}
        else:
            payload = {"items": workloads if argv[2] == "statefulsets" else []}
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=json.dumps(payload), stderr="")

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        soak_observer.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={BOOT_UUID_HEADER: "boot"})),
            **kwargs,
        ),
    )
    monkeypatch.setattr(soak_observer, "run_command", command)
    monkeypatch.setattr(progress, "run_command", command)

    failed = await observer.observe()
    recovered = await observer.observe()

    assert "progress" in failed.errors and failed.deployments == []
    assert "progress" not in recovered.errors
    assert len(recovered.deployments) == 1
    assert recovered.deployments[0].saved_iteration == 3
    assert recovered.deployments[0].finished_rollout_id == 4
