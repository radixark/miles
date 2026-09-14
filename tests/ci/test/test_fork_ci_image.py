import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_policy import resolve_workflow_inputs
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("fork_ci_image", ROOT / ".github/workflows/scripts/fork_ci_image.py")
HANDLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDLER)
REPOSITORY = "radixark/miles"


@pytest.fixture
def identity():
    request = dict(
        pr=3192,
        merge_sha="a" * 40,
        inputs_hash="b" * 64,
        run_id=123,
        run_attempt=2,
        force_rebuild=False,
        labels=["run-ci-image"],
    )
    base = dict(id=1, full_name=REPOSITORY)
    fork = dict(id=2, full_name="contributor/miles", owner=dict(login="contributor"))
    pr = dict(
        number=3192,
        state="open",
        base=dict(repo=base),
        head=dict(repo=fork, ref="image-fix", sha="c" * 40),
        labels=[dict(name="run-ci-image")],
    )
    run = dict(
        id=123,
        run_attempt=2,
        event="pull_request",
        status="in_progress",
        conclusion=None,
        created_at="2026-09-14T00:00:00Z",
        repository=base,
        head_repository=fork,
        head_branch="image-fix",
        head_sha="c" * 40,
        workflow_id=99,
        path=".github/workflows/pr-test.yml",
        pull_requests=[],
        referenced_workflows=[
            dict(
                path=f"{REPOSITORY}/.github/workflows/_build-pr-ci-image.yml@{'a' * 40}",
                ref="refs/pull/3192/merge",
                sha="a" * 40,
            )
        ],
    )
    return request, run, pr


def test_fork_with_empty_pr_array_is_bound_to_frozen_merge(identity):
    request, run, pr = identity
    pr["merge_commit_sha"] = "d" * 40
    HANDLER.validate_identity(request, run, pr, REPOSITORY, 99)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request, run, pr: request.update(run_attempt=1),
        lambda request, run, pr: request.update(run_id=124),
        lambda request, run, pr: request.update(pr=42),
        lambda request, run, pr: request.update(merge_sha="d" * 40),
        lambda request, run, pr: run.update(event="push"),
        lambda request, run, pr: run.update(workflow_id=42),
        lambda request, run, pr: run.update(path=".github/workflows/other.yml"),
        lambda request, run, pr: run.update(status="completed"),
        lambda request, run, pr: run.update(referenced_workflows=[]),
        lambda request, run, pr: pr.update(state="closed"),
        lambda request, run, pr: pr["head"].update(sha="d" * 40),
        lambda request, run, pr: pr["head"].update(ref="other"),
        lambda request, run, pr: pr["head"].update(repo=dict(id=3)),
    ],
)
def test_rejects_wrong_or_stale_identity(identity, mutation):
    request, run, pr = copy.deepcopy(identity)
    mutation(request, run, pr)
    with pytest.raises(ValueError):
        HANDLER.validate_identity(request, run, pr, REPOSITORY, 99)


def archive_request(request, filename="request.json"):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(filename, json.dumps(request))
    return data.getvalue()


def test_request_is_bounded_data(identity):
    request, _, _ = identity
    assert HANDLER.parse_request(archive_request(request)) == request
    for bad in [
        dict(request, pr=True),
        dict(request, force_rebuild="false"),
        dict(request, labels="run-ci-image"),
        dict(request, labels=[True]),
        dict(request, labels=None),
        dict(request, merge_sha="--upload-pack=evil"),
        dict(request, extra="script"),
        dict(request, inputs_hash="x" * 5000),
    ]:
        with pytest.raises(ValueError):
            HANDLER.parse_request(archive_request(bad))
    with pytest.raises(ValueError):
        HANDLER.parse_request(archive_request(request, "../request.json"))


def test_old_attempt_artifact_is_not_a_request(monkeypatch, identity):
    _, run, _ = identity
    monkeypatch.setattr(HANDLER, "api", lambda *args, **kwargs: [dict(name="fork-ci-image-123-1", expired=False)])
    assert HANDLER.resolve_request(dict(workflow_run=run), REPOSITORY) is None


def test_request_archive_cannot_select_another_run(monkeypatch, identity):
    request, run, _ = identity
    monkeypatch.setattr(
        HANDLER,
        "api",
        lambda *args, **kwargs: [dict(name="fork-ci-image-123-2", expired=False, size_in_bytes=1000, id=55)],
    )
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: archive_request(dict(request, run_id=124)))
    with pytest.raises(ValueError, match="another attempt"):
        HANDLER.resolve_request(dict(workflow_run=run), REPOSITORY)


def stub_api(monkeypatch, identity, *, cpu="success", latest=123):
    request, run, pr = identity
    calls = []

    def api(path, **kwargs):
        calls.append(path)
        if path.endswith("/actions/runs/123"):
            return run
        if path.endswith("/pulls/3192"):
            return pr
        if path.endswith("/actions/workflows/pr-test.yml"):
            return dict(id=99)
        if "/runs?" in path:
            return [dict(run, id=latest)]
        if "/jobs?filter=all" in path:
            return [dict(name=name, conclusion=cpu, run_attempt=1) for name in HANDLER.CPU_JOBS]
        raise AssertionError(path)

    monkeypatch.setattr(HANDLER, "api", api)
    return calls


def test_cpu_gate_retains_successful_jobs_on_partial_rerun(monkeypatch, identity):
    calls = stub_api(monkeypatch, identity)
    HANDLER.current_request(identity[0], REPOSITORY)
    assert any("/jobs?filter=all" in path for path in calls)
    stub_api(monkeypatch, identity, cpu="failure")
    with pytest.raises(ValueError, match="CPU A gate"):
        HANDLER.current_request(identity[0], REPOSITORY)
    identity[0]["labels"].append("bypass-fastfail")
    HANDLER.current_request(identity[0], REPOSITORY)


def test_partial_rerun_failure_overrides_old_cpu_success(monkeypatch, identity):
    stub_api(monkeypatch, identity)
    original = HANDLER.api

    def api(path, **kwargs):
        result = original(path, **kwargs)
        if "/jobs?" in path:
            result.insert(0, dict(name=next(iter(HANDLER.CPU_JOBS)), run_attempt=2, conclusion="failure"))
        return result

    monkeypatch.setattr(HANDLER, "api", api)
    with pytest.raises(ValueError, match="CPU A gate"):
        HANDLER.current_request(identity[0], REPOSITORY)


def test_rejects_superseded_runs(monkeypatch, identity):
    stub_api(monkeypatch, identity, latest=124)
    with pytest.raises(ValueError, match="superseded"):
        HANDLER.current_request(identity[0], REPOSITORY)


@pytest.fixture
def source(tmp_path, identity):
    root = tmp_path / "source"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (root / "docker").mkdir()
    (root / "docker/Dockerfile").write_text("FROM scratch\n")
    (root / "docker/build.py").write_text("raise RuntimeError('untrusted driver must not execute')\n")
    git("add", "docker")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (root / "docker/Dockerfile").write_text("FROM scratch\nLABEL pin=fixed\n")
    (root / "tests/e2e").mkdir(parents=True)
    (root / "tests/e2e/test_new.py").write_text(
        "raise RuntimeError('PR imports must not execute')\nregister_cuda_ci(est_time=1, suite='stage-b-2-gpu-h200', labels=['megatron'], hardware=['hopper'])\n"
    )
    git("add", "docker/Dockerfile", "tests")
    git("commit", "-qm", "fork")
    head = git("rev-parse", "HEAD")
    merge = git("commit-tree", "HEAD^{tree}", "-p", base, "-p", head, "-m", "merge")
    git("checkout", "-q", "--detach", merge)
    request, _, pr = copy.deepcopy(identity)
    request.update(merge_sha=merge, inputs_hash=HANDLER.image_inputs.compute("HEAD", root=root))
    pr["head"]["sha"] = head
    policy = resolve_workflow_inputs("pull_request", "", '["run-ci-image"]')
    return root, request, pr, policy


def test_validates_new_pr_registration_without_executing_source(source):
    root, request, pr, policy = source
    HANDLER.validate_source(request, pr, policy, root)
    for labels in ([], ["run-ci"], ["run-ci-unknown"], ["run-on-blackwell"]):
        cpu_policy = resolve_workflow_inputs("pull_request", "", json.dumps(labels))
        with pytest.raises(ValueError, match="No CUDA tests"):
            HANDLER.validate_source(request, pr, cpu_policy, root)


@pytest.mark.parametrize(
    "labels,cpu",
    [
        (["run-ci-image"], "success"),
        (["nightly"], "failure"),
        (["run-ci-image", "bypass-fastfail"], "failure"),
    ],
)
def test_label_removal_preserves_source_selection_and_cpu_gate(monkeypatch, identity, source, labels, cpu):
    root, request, pr, _ = source
    request["labels"] = labels
    pr["labels"] = []
    run = copy.deepcopy(identity[1])
    run["head_sha"] = pr["head"]["sha"]
    run["referenced_workflows"][0].update(
        path=f"{REPOSITORY}/.github/workflows/_build-pr-ci-image.yml@{request['merge_sha']}",
        sha=request["merge_sha"],
    )
    stub_api(monkeypatch, (request, run, pr), cpu=cpu)
    parsed = HANDLER.parse_request(archive_request(request))
    current_pr, fork_policy = HANDLER.current_request(parsed, REPOSITORY)
    caller_policy = resolve_workflow_inputs("pull_request", "", json.dumps(labels))
    assert fork_policy == caller_policy
    HANDLER.validate_source(parsed, current_pr, fork_policy, root)


def test_live_bypass_label_cannot_change_source_cpu_gate(monkeypatch, identity):
    identity[2]["labels"].append(dict(name="bypass-fastfail"))
    stub_api(monkeypatch, identity, cpu="failure")
    with pytest.raises(ValueError, match="CPU A gate"):
        HANDLER.current_request(identity[0], REPOSITORY)


@pytest.mark.parametrize(
    "path",
    [
        "docker/Dockerfile",
        "docker/build.py",
        ".dockerignore",
        "docker/Dockerfile.dockerignore",
        "tests/e2e/test_new.py",
    ],
)
def test_source_symlinks_cannot_reach_host_files(source, tmp_path, path):
    root, request, pr, policy = source
    target = tmp_path / "outside"
    target.write_text("host data")
    link = root / path
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    with pytest.raises(Exception, match="symlink"):
        HANDLER.validate_source(request, pr, policy, root)


def test_context_hash_must_match_frozen_source(source):
    root, request, pr, policy = source
    (root / "docker/Dockerfile").write_text("FROM scratch\nLABEL pin=other\n")
    with pytest.raises(ValueError, match="Build context differs"):
        HANDLER.validate_source(request, pr, policy, root)


def test_oci_build_uses_trusted_driver_and_supplied_context(source, monkeypatch):
    root, request, _, _ = source
    spec = importlib.util.spec_from_file_location("build", ROOT / "docker/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    commands = []
    monkeypatch.setattr(build, "run", lambda cmd, dry_run: commands.append(cmd))
    build.build_and_push(
        "cu13",
        "custom",
        False,
        "docker/Dockerfile",
        custom_tag="pr-3192",
        context=root,
        output="type=oci,dest=/tmp/output,tar=false",
    )
    command = commands[0]
    assert command[-1] == str(root)
    assert command[command.index("-f") + 1] == str(root / "docker/Dockerfile")
    assert command[command.index("--platform") + 1] == "linux/amd64,linux/arm64"
    assert command[command.index("--label") + 1] == f"miles.image-inputs={request['inputs_hash']}"
    assert "--push" not in command


def test_publication_requires_matching_hash_on_both_architectures(identity, monkeypatch):
    request, _, _ = identity
    config = dict(config=dict(Labels={"miles.image-inputs": request["inputs_hash"]}))
    manifest = {"linux/amd64": config, "linux/arm64": config}
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: json.dumps(manifest))
    HANDLER.verify_published(request)
    manifest["linux/arm64"] = dict(config=dict(Labels={"miles.image-inputs": "old"}))
    with pytest.raises(ValueError, match="linux/arm64"):
        HANDLER.verify_published(request)


@pytest.mark.parametrize("force,remove_fails", [(False, False), (True, False), (True, True)])
def test_publication_only_consumes_the_original_force_request(source, monkeypatch, capsys, force, remove_fails):
    root, request, pr, policy = source
    request["force_rebuild"] = force
    pr["labels"].append(dict(name="rebuild-ci-image"))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("REQUEST_JSON", json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["fork_ci_image.py", "finish", "--source", str(root)])
    monkeypatch.setattr(HANDLER, "current_request", lambda *args: (pr, policy))
    monkeypatch.setattr(HANDLER, "verify_published", lambda request: None)
    mutations = []

    def api(path, *, method):
        mutations.append((path, method))
        if remove_fails:
            raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(HANDLER, "api", api)
    HANDLER.main()
    assert mutations == ([(f"repos/{REPOSITORY}/issues/3192/labels/rebuild-ci-image", "DELETE")] if force else [])
    assert ("::warning::" in capsys.readouterr().out) == remove_fails


def test_observer_waits_past_an_old_attempt_job(monkeypatch, identity):
    request, run, _ = identity
    responses = iter([None, request])
    sleeps = []
    monkeypatch.setattr(HANDLER, "resolve_request", lambda *args: next(responses))
    monkeypatch.setattr(HANDLER.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        HANDLER,
        "api",
        lambda path, **kwargs: (
            [dict(name="docker-build / docker-build", run_attempt=1, status="completed")] if "/jobs?" in path else run
        ),
    )
    assert HANDLER.wait_request(dict(workflow_run=run), REPOSITORY) == request
    assert sleeps == [30]


@pytest.mark.parametrize("ended", ["completed", "new-attempt", "image-done"])
def test_observer_stops_without_a_request(monkeypatch, identity, ended):
    _, run, _ = identity
    current = dict(run)
    if ended == "completed":
        current["status"] = "completed"
    elif ended == "new-attempt":
        current["run_attempt"] += 1
    monkeypatch.setattr(HANDLER, "resolve_request", lambda *args: None)
    monkeypatch.setattr(
        HANDLER,
        "api",
        lambda path, **kwargs: (
            [dict(name="docker-build / docker-build", run_attempt=2, status="completed")]
            if "/jobs?" in path
            else current
        ),
    )
    monkeypatch.setattr(HANDLER.time, "sleep", lambda seconds: pytest.fail("No request can arrive"))
    assert HANDLER.wait_request(dict(workflow_run=run), REPOSITORY) is None


def publisher_run(**updates):
    return dict(
        dict(
            id=999,
            workflow_id=100,
            event="workflow_run",
            display_title="fork-ci-image-123-2",
            html_url="https://github.com/radixark/miles/actions/runs/999",
            status="completed",
            conclusion="success",
        ),
        **updates,
    )


def stub_publisher(monkeypatch, identity, *, listings, result=None, build="success"):
    _, source, _ = identity
    results = iter(listings)
    calls, sleeps = [], []

    def api(path, **kwargs):
        calls.append(path)
        if path.endswith("/actions/runs/123"):
            return source
        if path.endswith("/actions/workflows/build-fork-ci-image.yml"):
            return dict(id=100)
        if "/actions/workflows/100/runs?" in path:
            return next(results)
        if path.endswith("/actions/runs/999"):
            return result
        if "/actions/runs/999/jobs?" in path:
            return [dict(name="build", conclusion=build)]
        raise AssertionError(path)

    monkeypatch.setattr(HANDLER, "api", api)
    monkeypatch.setattr(HANDLER.time, "sleep", sleeps.append)
    return calls, sleeps


def test_waiter_correlates_publisher_and_waits_for_build_completion(monkeypatch, identity):
    request, _, _ = identity
    wrong = [
        publisher_run(display_title="fork-ci-image-123-1"),
        publisher_run(event="push"),
        publisher_run(workflow_id=101),
    ]
    calls, sleeps = stub_publisher(
        monkeypatch,
        identity,
        listings=[wrong, [publisher_run(status="in_progress", conclusion=None)]],
        result=publisher_run(),
    )
    HANDLER.wait_build(request, REPOSITORY)
    assert sleeps == [30, 30]
    assert any("/999/jobs?" in path for path in calls)
    assert all("/rerun" not in path for path in calls)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_waiter_propagates_publication_failure(monkeypatch, identity, conclusion):
    stub_publisher(monkeypatch, identity, listings=[[publisher_run(conclusion=conclusion)]])
    with pytest.raises(ValueError, match=conclusion):
        HANDLER.wait_build(identity[0], REPOSITORY)


def test_successful_workflow_without_a_build_cannot_unblock_gpu(monkeypatch, identity):
    stub_publisher(monkeypatch, identity, listings=[[publisher_run()]], build="skipped")
    with pytest.raises(ValueError, match="without a successful image build"):
        HANDLER.wait_build(identity[0], REPOSITORY)


@pytest.mark.parametrize("updates", [dict(status="completed"), dict(run_attempt=3)])
def test_waiter_rejects_an_inactive_source(monkeypatch, identity, updates):
    request, run, _ = identity
    run.update(updates)
    stub_publisher(monkeypatch, identity, listings=[])
    with pytest.raises(ValueError, match="no longer active"):
        HANDLER.wait_build(request, REPOSITORY)


def test_source_cancellation_stops_the_build_process_group(monkeypatch, identity, tmp_path):
    trusted = tmp_path / "trusted"
    (trusted / "docker").mkdir(parents=True)
    child_pid = tmp_path / "child.pid"
    child_code = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    (trusted / "docker/build.py").write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(HANDLER, "ROOT", trusted)
    monkeypatch.setenv("OCI_OUTPUT", str(tmp_path / "oci"))
    checks = []

    def active(*args):
        checks.append(True)
        if len(checks) > 1:
            deadline = time.monotonic() + 5
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child_pid.exists(), "Build child did not start"
            raise ValueError("Source CI attempt is no longer active")

    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        wait = process.wait
        process.wait = lambda timeout=None: wait(timeout=0.05 if timeout == 30 else timeout)
        return process

    monkeypatch.setattr(HANDLER, "require_active_source", active)
    monkeypatch.setattr(HANDLER.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="no longer active"):
        HANDLER.build_image(identity[0], REPOSITORY, tmp_path)
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, 9)
        pytest.fail("Build child survived source cancellation")


def test_workflow_preserves_gate_and_build_publish_boundary():
    parent = yaml.safe_load((ROOT / ".github/workflows/pr-test.yml").read_text())
    partitions = parent["jobs"]["stage-a-cpu"]["strategy"]["matrix"]["partition_id"]
    assert HANDLER.CPU_JOBS == {f"stage-a-cpu ({part}) / run-cpu" for part in partitions}
    reusable = yaml.safe_load((ROOT / ".github/workflows/_build-pr-ci-image.yml").read_text())
    image_job = reusable["jobs"]["docker-build"]
    assert "head.repo.full_name == github.repository" in image_job["runs-on"]
    assert "steps.wait.outputs.built" in image_job["outputs"]["built"]
    assert "steps.wait.outputs.built == 'true'" in image_job["outputs"]["tag_available"]
    assert parent["jobs"]["docker-build"]["permissions"]["actions"] == "read"
    wait = next(step for step in image_job["steps"] if step.get("id") == "wait")
    assert "wait-build" in wait["run"] and 'echo "built=true"' in wait["run"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/build-fork-ci-image.yml").read_text())
    assert workflow[True]["workflow_run"]["types"] == ["in_progress"]
    assert workflow["jobs"]["build"]["permissions"]["actions"] == "read"
    assert workflow["jobs"]["build"]["concurrency"]["cancel-in-progress"] is True
    steps = workflow["jobs"]["build"]["steps"]
    cleanup = next(
        step for step in steps if step.get("name") == "Remove the builder before introducing publishing credentials"
    )
    assert "always()" in cleanup["if"]
    names = [step.get("name", "") for step in steps]
    assert (
        names.index("Build without registry credentials")
        < names.index("Remove the builder before introducing publishing credentials")
        < names.index("Login to Docker Hub for publication")
    )
    buildx = next(step for step in steps if step.get("id") == "buildx")
    assert buildx["with"]["buildkitd-flags"] == "--oci-worker-net=bridge"
    assert "network=host" not in buildx["with"]["driver-opts"]
    assert all(
        step["with"]["persist-credentials"] is False
        for step in steps
        if step.get("uses", "").startswith("actions/checkout")
    )
