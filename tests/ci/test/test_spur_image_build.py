"""Exercise the Spur image-build boundary without a cluster or registry."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci
from tests.ci.test import test_fork_ci_image as fork_tests

HANDLER = fork_tests.HANDLER
REPOSITORY = fork_tests.REPOSITORY
identity = fork_tests.identity
source = fork_tests.source
rocm_upgrade_source = fork_tests.rocm_upgrade_source

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("spur_build", ROOT / ".github/workflows/scripts/spur/build_image.py")
SPUR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPUR)


@pytest.fixture
def environment(monkeypatch, tmp_path):
    output = tmp_path / "shared" / "oci"
    output.mkdir(parents=True)
    for key, value in {
        "OCI_OUTPUT": str(output),
        "CI_NODE_SCRATCH": str(tmp_path),
        "SLURM_PARTITION": "ci",
        "SLURM_ACCOUNT": "miles",
        "SLURM_QOS": "normal",
        "DOCKERHUB_TOKEN": "write-secret",
        "GH_TOKEN": "github-secret",
        "HF_TOKEN": "hf-secret",
        "BUILD_GITHUB_TOKEN": "build-secret",
        "BASH_ENV": "/must/not/source",
        "HTTP_PROXY": "https://user:proxy-secret@proxy",
        "SLURM_EXPORT_ENV": "ALL",
    }.items():
        monkeypatch.setenv(key, value)
    return output.parent


def test_rocm_same_repository_request_uses_the_trusted_boundary(identity):
    request, run, pr = identity
    run["head_repository"] = pr["head"]["repo"] = pr["base"]["repo"]
    with pytest.raises(ValueError, match="Expected a fork PR"):
        HANDLER.validate_identity(request, run, pr, REPOSITORY, 99)
    request["variant"] = "rocm10-mi35x-pr"
    run["path"] = ".github/workflows/pr-test-rocm.yml"
    HANDLER.validate_identity(request, run, pr, REPOSITORY, 99)


def test_rocm_handoff_carries_literals_and_frozen_identity(rocm_upgrade_source, monkeypatch, environment):
    root, request, pr, sentinel, base, wheels, _ = rocm_upgrade_source
    HANDLER.validate_source(request, pr, root)
    seen = []

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(HANDLER, "require_active_source", lambda *args: None)

    def popen(command, **kwargs):
        assert "BUILD_GITHUB_TOKEN" not in kwargs["env"]
        seen.append(command)
        return Process()

    monkeypatch.setattr(HANDLER.subprocess, "Popen", popen)
    HANDLER.build_image(request, REPOSITORY, root)
    command = seen[0]
    assert command[1:3] == [str(ROOT / ".github/workflows/scripts/spur/build_image.py"), "launch"]
    assert command[command.index("--merge-sha") + 1] == request["merge_sha"]
    assert command[command.index("--inputs-hash") + 1] == request["inputs_hash"]
    build = json.loads(command[command.index("--command") + 1])
    assert f"SGLANG_IMAGE_TAG={base}" in build and f"WHEELS_TAG_ROCM={wheels}" in build
    assert build[1] == str(ROOT / "docker/build.py")
    assert not sentinel.exists()


@pytest.mark.parametrize("failure", [False, True])
def test_submission_scrubs_secrets_and_cancels_its_exact_job(monkeypatch, environment, failure):
    calls = []
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)

    def output(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"] == SPUR.scheduler_env()
        assert set(kwargs["env"]) <= {"PATH", "SLURM_CONF"}
        if command[0] == "sbatch":
            assert "--exclusive" in command and "--export=ALL" in command
            assert command[command.index("--exclude") + 1] == SPUR.EXCLUDE
            assert not any("write-secret" in arg or "github-secret" in arg for arg in command)
            return "4321\n"
        return "RUNNING\n" if failure and not any(call[0][0] == "scancel" for call in calls) else ""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    def wait(*args):
        if failure:
            raise RuntimeError("source ended")

    monkeypatch.setattr(SPUR.subprocess, "check_output", output)
    monkeypatch.setattr(SPUR.subprocess, "run", run)
    monkeypatch.setattr(SPUR, "wait_for_job", wait)
    monkeypatch.setattr(SPUR.signal, "signal", lambda *args: None)
    if failure:
        with pytest.raises(RuntimeError, match="source ended"):
            SPUR.launch(Path("/shared/source"), "a" * 40, "b" * 64, ["trusted-command"])
        assert [call[0] for call in calls if call[0][0] == "scancel"] == [["scancel", "4321"]]
    else:
        SPUR.launch(Path("/shared/source"), "a" * 40, "b" * 64, ["trusted-command"])
    assert not (environment / "exit-code").exists()


def test_cancel_recovers_a_submission_whose_reply_was_interrupted(monkeypatch, environment):
    calls = []
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)

    def output(command, **kwargs):
        if "--name" in command:
            assert command[command.index("--name") + 1] == f"miles-pr-image-{environment.name}"
            return "4321\n"
        return "RUNNING\n" if not calls else ""

    monkeypatch.setattr(SPUR.subprocess, "check_output", output)
    monkeypatch.setattr(SPUR.subprocess, "run", lambda command, **kwargs: calls.append(command))
    SPUR.cancel(environment)
    assert calls == [["scancel", "4321"]]


@pytest.mark.parametrize("states", [["", "RUNNING", "", ""], ["RUNNING", "", "RUNNING", "", ""]])
def test_cancel_always_signals_and_requires_two_consecutive_empty_reads(monkeypatch, environment, states):
    # The review's two counterexamples: a transient empty before cancellation,
    # and another between two RUNNING observations after cancellation.
    (environment / "job-id").write_text("4321")
    observations = iter(states)
    calls = []
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)

    def output(command, **kwargs):
        assert calls and calls[0] == ["scancel", "4321"]
        value = next(observations)
        calls.append(["queue", value])
        return value

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(SPUR.subprocess, "check_output", output)
    monkeypatch.setattr(SPUR.subprocess, "run", run)
    SPUR.cancel(environment)
    assert [call[1] for call in calls if call[0] == "queue"] == states
    assert next(observations, "exhausted") == "exhausted"


def test_name_recovery_retries_a_transient_empty_response(monkeypatch, environment):
    responses = iter(["", "4321", "", ""])
    commands = []
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)

    def output(command, **kwargs):
        commands.append(command)
        return next(responses)

    monkeypatch.setattr(SPUR.subprocess, "check_output", output)
    monkeypatch.setattr(
        SPUR.subprocess,
        "run",
        lambda command, **kwargs: (commands.append(command) or subprocess.CompletedProcess(command, 0)),
    )
    SPUR.cancel(environment)
    assert len([command for command in commands if "--name" in command]) == 2
    assert ["scancel", "4321"] in commands
    assert next(responses, "exhausted") == "exhausted"


def test_completed_job_accepts_scancel_error_only_after_confirmed_absence(monkeypatch):
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)
    monkeypatch.setattr(SPUR.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    monkeypatch.setattr(SPUR.subprocess, "check_output", lambda *args, **kwargs: "")
    SPUR.stop_job("4321")
    monkeypatch.setattr(SPUR.subprocess, "check_output", lambda *args, **kwargs: "RUNNING")
    with pytest.raises(subprocess.CalledProcessError):
        SPUR.stop_job("4321")


@pytest.mark.parametrize("cancel_failed", [False, True])
def test_workflow_retains_shared_files_until_cancellation_is_confirmed(tmp_path, cancel_failed):
    shared, auth = tmp_path / "shared", tmp_path / "auth"
    shared.mkdir()
    auth.mkdir()
    binary = tmp_path / "python3"
    binary.write_text(f"#!/bin/sh\nexit {int(cancel_failed)}\n")
    binary.chmod(0o700)
    steps = yaml.safe_load((ROOT / ".github/workflows/build-fork-ci-image.yml").read_text())["jobs"]["build"]["steps"]
    result = subprocess.run(
        ["bash", "-e", "-c", steps[-1]["run"]],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "DOCKER_CONFIG": str(auth),
            "IMAGE_RUN_DIR": str(shared),
            "IMAGE_VARIANT": "rocm10-mi35x-pr",
        },
    )
    assert shared.exists() is cancel_failed
    assert (result.returncode != 0) is cancel_failed
    assert not auth.exists()


@pytest.mark.parametrize("marker", [None, "1", "0"])
def test_missing_or_failed_cluster_completion_blocks_publication(monkeypatch, tmp_path, marker):
    monkeypatch.setattr(SPUR, "queued", lambda job_id: False)
    monkeypatch.setattr(SPUR.time, "sleep", lambda *args: None)
    if marker is not None:
        (tmp_path / "exit-code").write_text(marker)
    if marker == "0":
        SPUR.wait_for_job("4321", tmp_path)
    else:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            SPUR.wait_for_job("4321", tmp_path)


@pytest.mark.parametrize("failure", ["none", "build", "cleanup"])
@pytest.mark.parametrize("rocm_upgrade_source", [True], indirect=True)
def test_compute_build_runs_trusted_cli_without_credentials_or_devices(
    rocm_upgrade_source, monkeypatch, environment, tmp_path, failure
):
    root, request, pr, sentinel, base, wheels, _ = rocm_upgrade_source
    HANDLER.validate_source(request, pr, root)
    command = HANDLER.build_command(request, root)
    (environment / "build.json").write_text(
        json.dumps(
            dict(
                source=str(root),
                merge_sha=request["merge_sha"],
                inputs_hash=request["inputs_hash"],
                command=command,
                scratch=str(tmp_path),
                trusted=str(ROOT),
            )
        )
    )
    real_run = subprocess.run
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "env" not in kwargs:
            return real_run(argv, **kwargs)
        env = kwargs["env"]
        assert not {"DOCKERHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "BUILD_GITHUB_TOKEN", "HTTP_PROXY"} & env.keys()
        assert env["DOCKER_CONFIG"].startswith(str(tmp_path))
        if argv[0] == "/usr/bin/python3" and "venv" in argv:
            return subprocess.CompletedProcess(argv, 0)
        if "pip" in argv:
            return subprocess.CompletedProcess(argv, 0)
        if argv[0] == "docker":
            code = int(failure == "cleanup" and "rm" in argv)
            return subprocess.CompletedProcess(argv, code)
        if str(argv[1]).endswith("docker/build.py"):
            # Run the real trusted driver/typer; intercept its Docker child only.
            assert str(argv[1]).startswith(str(tmp_path))
            binary = Path(env["HOME"]) / "docker"
            binary.write_text(
                f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(tmp_path / 'argv')!r}).write_text(json.dumps(sys.argv[1:]))\nsys.exit({int(failure == 'build')})\n"
            )
            binary.chmod(0o700)
            argv[0] = sys.executable
            return real_run(argv, **{**kwargs, "env": {**env, "PATH": str(binary.parent) + ":" + env["PATH"]}})
        return real_run(argv, **kwargs)

    monkeypatch.setattr(SPUR.subprocess, "run", run)
    monkeypatch.setattr(SPUR.signal, "signal", lambda *args: None)
    if failure == "build":
        with pytest.raises(subprocess.CalledProcessError):
            SPUR.node(environment)
    else:
        SPUR.node(environment)
    assert (environment / "exit-code").read_text() == ("0" if failure == "none" else "1")
    assert not sentinel.exists()
    argv = json.loads((tmp_path / "argv").read_text())
    assert f"SGLANG_IMAGE_TAG={base}" in argv and f"WHEELS_TAG_ROCM={wheels}" in argv
    assert f"miles.image-inputs={request['inputs_hash']}" in argv
    assert "radixark/miles:pr-3192-rocm10-mi35x" in argv
    assert not any(arg in argv for arg in ("--push", "--secret", "--allow", "--device"))
    assert any(call[0][:3] == ["docker", "buildx", "rm"] for call in calls)
    assert not list(tmp_path.glob("miles-pr-image-*"))


def test_workflow_keeps_rocm_off_docker_build_and_secrets_after_build():
    caller = yaml.safe_load((ROOT / ".github/workflows/_build-pr-ci-image.yml").read_text())["jobs"]["docker-build"]
    assert "inputs.variant == 'cu13'" in caller["runs-on"]
    assert "inputs.variant == 'rocm10-mi35x-pr'" in caller["env"]["REQUEST_BUILD"]
    publisher = yaml.safe_load((ROOT / ".github/workflows/build-fork-ci-image.yml").read_text())["jobs"]["build"]
    assert "linux-miles-mi35x-spur" in publisher["runs-on"]
    steps = publisher["steps"]
    build = next(i for i, step in enumerate(steps) if step.get("name") == "Build without registry credentials")
    login = next(i for i, step in enumerate(steps) if step.get("name") == "Login to Docker Hub for publication")
    assert build < login
    for step in steps[:login]:
        assert "secrets.DOCKERHUB" not in str(step)
    paths = next(step for step in steps if step.get("id") == "paths")["run"]
    assert "mktemp -d /tmp/miles-image-auth." in paths
    assert "cancel --run-dir" in steps[-1]["run"]
