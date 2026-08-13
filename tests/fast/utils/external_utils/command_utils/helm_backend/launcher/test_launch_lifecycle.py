import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import yaml
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import cluster_info, log_follower
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.workers.k8s_types import ContainerState, ContainerStatus, Pod, PodMetadata, PodStatus


def _write(path: Path, status: OrchestratorStatus, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False) -> None:
    monkeypatch.setattr(entrypoint, "compute_specs", lambda args: specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(colocate=colocate, argv=[], use_wandb=False, wandb_run_id=None),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestPerLaunchExitFile:
    def test_every_launch_token_names_a_file_of_its_own(self):
        """A relaunch that reused the path could read the last words of the wrapper it just replaced."""
        first = RunFiles.new_state_file(run_directory="/runs/abc")
        second = RunFiles.new_state_file(run_directory="/runs/abc")

        assert first != second
        assert first.parent == second.parent == Path("/runs/abc/state")

    def test_the_verdict_of_another_launch_is_never_read(self, tmp_path):
        """The wrapper of the run being replaced writes its own file, which this launcher must not consult."""
        mine = _orchestrator_state_path(tmp_path, "a")
        theirs = _orchestrator_state_path(tmp_path, "b")
        _write(theirs, OrchestratorStatus.EXITED, exit_code=7)

        assert (
            observer._compute_run_outcome(state=OrchestratorState.read(mine), phase="Running", missing_polls=0) is None
        )

    def test_a_verdict_written_before_the_launcher_looked_is_returned_at_once(self, tmp_path):
        """Relaunching a finished run attaches to it, and its exit code is already sitting in the file."""
        state_file = _orchestrator_state_path(tmp_path, "a")
        _write(state_file, OrchestratorStatus.EXITED, exit_code=4)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(state_file), phase="Running", missing_polls=0
        )

        assert outcome.exit_code == 4


class TestBackgroundLogOrdering:
    def test_the_watcher_runs_while_the_logs_stream(self, monkeypatch, tmp_path):
        """kubectl logs --follow never returns on a keep-alive pod, so a serial launcher would never poll."""
        state_file = tmp_path / "orchestrator.state"
        _write(state_file, OrchestratorStatus.EXITED, exit_code=3)
        started: list[str] = []

        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False
                self.stdout = iter(())

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def poll(self) -> int | None:
                return 0 if self.terminated else None

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.terminated = True

        process = FakeProcess()

        def fake_popen(command, *args, **kwargs):
            started.append(command[0])
            return process

        pods = [_pod()]
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: pods)
        monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(entrypoint, "pod_phase", lambda namespace, workload: "Running")
        with pytest.raises(SystemExit) as raised:
            entrypoint._follow_until_finished(release="myrun", namespace="myns", state_file=state_file)

        assert raised.value.code == 3
        assert started and set(started) == {"kubectl"}


def _pod() -> Pod:
    container = ContainerStatus(
        name="orchestrator", container_id="docker://a", restart_count=0, state=ContainerState(running={})
    )
    return Pod(
        metadata=PodMetadata(name="p", uid="p-uid"),
        status=PodStatus(phase="Running", container_statuses=[container]),
    )


class _Recorded(NamedTuple):
    kubectl: list[list[str]]
    upgraded: list[str]


class TestAutoUninstallValues:
    def test_the_launcher_leaves_the_self_uninstall_decision_to_the_chart(self, monkeypatch, tmp_path):
        """The chart default arms it and a user values file may override it, so the launcher writes nothing."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        assert "autoUninstall" not in _written_values(tmp_path)["run"]


class TestDefusingAPendingUninstall:
    def test_a_first_install_deletes_a_job_an_earlier_run_of_this_id_left_behind(self, monkeypatch, tmp_path):
        """A relaunch of a run id whose uninstall is still sleeping would be uninstalled by that job."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        assert _DELETE_UNINSTALL_JOB in recorded.kubectl
        assert recorded.upgraded == [_RELEASE]

    def test_a_relaunch_that_would_change_more_than_the_size_installs_nothing(self, monkeypatch, tmp_path):
        """A refused launch must leave the running one alone, including the job that would uninstall it."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        with pytest.raises(SystemExit, match="more than its size"):
            _launch(monkeypatch, tmp_path, recorded, installed=True, proposed_differs=True)

        assert recorded.upgraded == []
        assert recorded.kubectl == []

    def test_attaching_to_the_orchestrator_already_installed_leaves_the_job_alone(self, monkeypatch, tmp_path):
        """No wrapper restarts here, so a deleted job is never recreated and the release would leak forever."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=True)

        assert recorded.kubectl == []
        assert recorded.upgraded == [_RELEASE]

    def test_a_delete_the_cluster_refused_stops_the_launch(self, monkeypatch, tmp_path):
        """Installing over a job that is still armed hands the new run's release to the old run's uninstall."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        with pytest.raises(subprocess.CalledProcessError):
            _launch(monkeypatch, tmp_path, recorded, installed=False, delete_fails=True)

        assert recorded.upgraded == []


_RUN_ID = "260101-000000-000"
_RELEASE = f"miles-run-{_RUN_ID}"
_DELETE_UNINSTALL_JOB = [
    "kubectl",
    "delete",
    "job",
    f"{_RELEASE}-uninstall",
    "--namespace",
    "rl",
    "--ignore-not-found",
]


class TestTheRelaunchKeepsThePodsItAttachesTo:
    def test_the_pods_keep_the_record_of_the_launch_that_installed_them(self, monkeypatch, tmp_path):
        """A new record in the pod template is a new pod template, so a resize would restart every worker."""
        recorded = _Recorded()

        _launch(monkeypatch, tmp_path, recorded, installed=True, rendered=_RENDERED_WITH_RECORD)

        assert _written_values(tmp_path)["run"]["launchRecord"] == _INSTALLED_RECORD

    def test_a_first_install_points_the_pods_at_the_record_of_this_launch(self, monkeypatch, tmp_path):
        recorded = _Recorded()

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        written = sorted((tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "launches").glob("*"))
        assert _written_values(tmp_path)["run"]["launchRecord"] == str(written[0])

    def test_this_launch_is_still_recorded_on_disk_when_the_pods_keep_an_older_one(self, monkeypatch, tmp_path):
        """The pods keeping their own record must not cost the run the record of this invocation."""
        recorded = _Recorded()

        _launch(monkeypatch, tmp_path, recorded, installed=True, rendered=_RENDERED_WITH_RECORD)

        written = sorted((tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "launches").glob("*"))
        assert len(written) == 1
        assert json.loads(written[0].read_text())["run_id"] == _RUN_ID


def _launch(
    monkeypatch,
    tmp_path,
    recorded: _Recorded,
    *,
    installed: bool,
    proposed_differs: bool = False,
    delete_fails: bool = False,
    rendered: str | None = None,
) -> None:
    def fake_run_process(command, **kwargs):
        arguments = [str(part) for part in command]
        recorded.kubectl.append(arguments)
        result = subprocess.CompletedProcess(
            args=command, returncode=1 if delete_fails else 0, stdout="", stderr="the api server refused"
        )
        if kwargs.get("check") and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, stderr=result.stderr)
        return result

    monkeypatch.setattr(command_wrapper, "run_process", fake_run_process)
    monkeypatch.setattr(Helm, "build_dependencies", staticmethod(lambda chart: None))
    rendered = rendered if rendered is not None else _RENDERED
    monkeypatch.setattr(
        Helm,
        "get_manifest",
        staticmethod(lambda release, namespace: Manifest.parse(rendered, namespace=namespace) if installed else None),
    )
    proposed = _RENDERED_WITH_ANOTHER_KEY if proposed_differs else rendered
    monkeypatch.setattr(
        Helm,
        "render_upgrade",
        staticmethod(lambda **kwargs: Manifest.parse(proposed, namespace=kwargs["namespace"])),
    )
    monkeypatch.setattr(Helm, "upgrade", staticmethod(lambda **kwargs: recorded.upgraded.append(kwargs["release"])))
    monkeypatch.setattr(Manifest, "state_file", lambda self, container: tmp_path / "attached.state")
    monkeypatch.setattr(entrypoint, "repo_base_dir", str(REPO_ROOT))

    _stub_launch_inputs(monkeypatch, specs=[])

    return entrypoint.execute_train(
        request=_train_request(),
        config=ExecuteTrainConfig(
            namespace="rl", run_id=_RUN_ID, helm_values=(str(_launchable_infra_file(tmp_path)),)
        ),
    )


def _written_values(tmp_path) -> dict:
    written = sorted(
        (tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "values").glob("values-*.yaml")
    )
    assert len(written) == 1, written
    return yaml.safe_load(written[0].read_text())


def _launchable_infra_file(tmp_path) -> Path:
    path = tmp_path / "launchable-infra.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "infra": {
                    "image": {"repository": "registry.local/miles", "tag": "v1"},
                    "sharedStorage": {
                        "type": "hostPath",
                        "hostPath": str(tmp_path / "cluster-storage"),
                        "mountPath": str(tmp_path / "cluster-storage"),
                    },
                }
            }
        )
    )
    return path


_RENDERED = "---\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: myrun-values\n"
_INSTALLED_RECORD = "/shared/miles-runs/the-launch-that-installed-these-pods/launches/launch-1.json"
_RENDERED_WITH_RECORD = f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: myrun-orchestrator
spec:
  template:
    spec:
      containers:
        - name: orchestrator
          command: [python]
          env:
            - name: MILES_SCRIPT_ENV_REPORT
              value: '{_INSTALLED_RECORD}'
"""
_RENDERED_WITH_ANOTHER_KEY = _RENDERED + "data:\n  extra: added\n"


def _train_request() -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args="",
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
    )
