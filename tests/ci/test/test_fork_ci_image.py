import copy
import importlib.util
import io
import json
import subprocess
import sys
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
    request = dict(pr=3192, merge_sha="a" * 40, inputs_hash="b" * 64, run_id=123, run_attempt=2)
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
        status="completed",
        conclusion="failure",
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
        lambda request, run, pr: run.update(conclusion="success"),
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
        if "/attempts/2/jobs?" in path:
            return [dict(name=name, conclusion=cpu) for name in HANDLER.CPU_JOBS]
        raise AssertionError(path)

    monkeypatch.setattr(HANDLER, "api", api)
    return calls


def test_requires_cpu_gate_from_exact_attempt(monkeypatch, identity):
    calls = stub_api(monkeypatch, identity)
    HANDLER.current_request(identity[0], REPOSITORY)
    assert any("/attempts/2/jobs?" in path for path in calls)
    stub_api(monkeypatch, identity, cpu="failure")
    with pytest.raises(ValueError, match="CPU A gate"):
        HANDLER.current_request(identity[0], REPOSITORY)
    identity[2]["labels"].append(dict(name="bypass-fastfail"))
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
    monkeypatch.setattr(HANDLER.image_inputs, "inspect_published", lambda image: json.dumps(manifest))
    HANDLER.verify_published(request)
    manifest["linux/arm64"] = dict(config=dict(Labels={"miles.image-inputs": "old"}))
    with pytest.raises(ValueError, match="linux/arm64"):
        HANDLER.verify_published(request)


def test_failed_label_consumption_cannot_rerun_forever(source, monkeypatch):
    root, request, pr, policy = source
    pr["labels"].append(dict(name="rebuild-ci-image"))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("REQUEST_JSON", json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["fork_ci_image.py", "finish", "--source", str(root)])
    monkeypatch.setattr(HANDLER, "current_request", lambda *args: (pr, policy))
    monkeypatch.setattr(HANDLER, "verify_published", lambda request: None)
    mutations = []

    def api(path, *, method):
        mutations.append((path, method))
        raise RuntimeError("label removal failed")

    monkeypatch.setattr(HANDLER, "api", api)
    with pytest.raises(RuntimeError, match="label removal failed"):
        HANDLER.main()
    assert len(mutations) == 1 and mutations[0][1] == "DELETE"


def test_publication_reruns_the_entire_workflow(source, monkeypatch):
    root, request, pr, policy = source
    pr["labels"].append(dict(name="rebuild-ci-image"))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("REQUEST_JSON", json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["fork_ci_image.py", "finish", "--source", str(root)])
    monkeypatch.setattr(HANDLER, "current_request", lambda *args: (pr, policy))
    monkeypatch.setattr(HANDLER, "verify_published", lambda request: None)
    mutations = []
    monkeypatch.setattr(HANDLER, "api", lambda path, **kwargs: mutations.append((path, kwargs["method"])))
    HANDLER.main()
    assert mutations == [
        (f"repos/{REPOSITORY}/issues/3192/labels/rebuild-ci-image", "DELETE"),
        (f"repos/{REPOSITORY}/actions/runs/123/rerun", "POST"),
    ]


def test_matching_published_image_cannot_start_another_build(source, monkeypatch):
    root, request, pr, policy = source
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("REQUEST_JSON", json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["fork_ci_image.py", "check-source", "--source", str(root)])
    monkeypatch.setattr(HANDLER, "current_request", lambda *args: (pr, policy))
    manifest = dict(config=dict(Labels={"miles.image-inputs": request["inputs_hash"]}))
    monkeypatch.setattr(HANDLER.image_inputs, "inspect_published", lambda image: json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing a repeated build request"):
        HANDLER.main()


def test_workflow_preserves_gate_and_build_publish_boundary():
    parent = yaml.safe_load((ROOT / ".github/workflows/pr-test.yml").read_text())
    partitions = parent["jobs"]["stage-a-cpu"]["strategy"]["matrix"]["partition_id"]
    assert HANDLER.CPU_JOBS == {f"stage-a-cpu ({part}) / run-cpu" for part in partitions}
    reusable = yaml.safe_load((ROOT / ".github/workflows/_build-pr-ci-image.yml").read_text())
    assert "head.repo.full_name == github.repository" in reusable["jobs"]["docker-build"]["runs-on"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/build-fork-ci-image.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
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
