"""Exercise PR image eligibility and the published-image handoff on CPU."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
import yaml
from tests.ci import rocm
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("rocm_image_inputs", ROOT / "docker/image_inputs.py")
IMAGE_INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGE_INPUTS)
VARIANT = "rocm10-mi35x-pr"
LOOKUP_SPEC = importlib.util.spec_from_file_location(
    "rocm_pr_lookup", ROOT / ".github/workflows/scripts/rocm_pr_image.py"
)
LOOKUP = importlib.util.module_from_spec(LOOKUP_SPEC)
LOOKUP_SPEC.loader.exec_module(LOOKUP)


@pytest.mark.parametrize("status", [200, 404, 401, 403, 429, 500])
def test_public_tag_lookup_distinguishes_absence_from_errors(monkeypatch, status):
    def open_url(request, *, timeout):
        assert request.method == "HEAD"
        assert (
            request.full_url
            == "https://hub.docker.com/v2/namespaces/radixark/repositories/miles/tags/pr-42-rocm10-mi35x"
        )
        assert not request.has_header("Authorization")
        assert timeout == 30
        if status != 200:
            raise HTTPError(request.full_url, status, "fixture", {}, None)
        from contextlib import nullcontext
        from types import SimpleNamespace

        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(LOOKUP, "urlopen", open_url)
    if status in (200, 404):
        assert LOOKUP.published_tag("42") == ("pr-42-rocm10-mi35x" if status == 200 else "")
    else:
        with pytest.raises(HTTPError):
            LOOKUP.published_tag("42")


@pytest.mark.parametrize("error", [URLError("network unavailable"), TimeoutError("timeout")])
def test_public_lookup_network_errors_fail_without_defaulting(monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(LOOKUP, "urlopen", fail)
    monkeypatch.setattr(sys, "argv", ["rocm_pr_image.py", "42"])
    assert LOOKUP.main() == 1
    assert capsys.readouterr().out == ""


def test_rocm_file_lookup_withholds_hosted_login_and_propagates_errors(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-ci-file.yml").read_text())
    steps = workflow["jobs"]["resolve-file-run"]["steps"]
    login = next(step for step in steps if step.get("name") == "Login to Docker Hub")
    assert login["if"] == "inputs.hw != 'rocm' && inputs.ci_image_tag == ''"
    step = next(step for step in steps if step.get("id") == "image")
    binary = tmp_path / "python3"
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CI_HW": "rocm",
            "CI_IMAGE_TAG": "",
            "PULL_NUMBER": "42",
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert result.returncode != 0
    assert not output.exists()


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    files = [
        ".dockerignore",
        "docker/build.py",
        "docker/image_inputs.py",
        "docker/install-kube-tools.sh",
        "docker/Dockerfile",
        "docker/Dockerfile.rubin",
        "docker/Dockerfile.rocm",
        "docker/Dockerfile.tpu",
        "docker/verify_transformer_engine.py",
        "docker/tpu_patch/fix.patch",
        "docker/amd_patch/latest/fix.patch",
        "docker/amd/install.py",
        "docker/patch/cuda.patch",
        "requirements.txt",
        "source.py",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base\n")
    git("add", ".")
    git("commit", "-qm", "base")
    return tmp_path, git


@pytest.mark.parametrize(
    "path,variants",
    [
        (".dockerignore", None),
        ("docker/build.py", None),
        ("docker/image_inputs.py", None),
        ("docker/install-kube-tools.sh", None),
        ("requirements.txt", None),
        ("docker/Dockerfile.rocm", [VARIANT]),
        ("docker/amd_patch/latest/fix.patch", [VARIANT]),
        ("docker/amd/install.py", [VARIANT]),
        ("docker/Dockerfile", ["cu13", "rubin"]),
        ("docker/Dockerfile.rubin", ["cu13", "rubin"]),
        ("docker/verify_transformer_engine.py", ["cu13", "rubin"]),
        ("docker/patch/cuda.patch", ["cu13", "rubin"]),
        ("docker/Dockerfile.tpu", ["tpu"]),
        ("docker/tpu_patch/fix.patch", ["tpu"]),
        ("source.py", []),
    ],
)
def test_input_mapping_separates_common_and_hardware_paths(path, variants):
    for variant in ("cu13", "rubin", VARIANT, "tpu"):
        assert IMAGE_INPUTS._matches(path, variant) is (variants is None or variant in variants)


@pytest.mark.parametrize(
    "variant,path,eligible",
    [
        ("cu13", "requirements.txt", True),
        (VARIANT, "requirements.txt", True),
        ("tpu", "requirements.txt", True),
        (VARIANT, "docker/Dockerfile.rocm", True),
        ("cu13", "docker/Dockerfile.rocm", False),
        (VARIANT, "docker/Dockerfile", False),
        ("cu13", "docker/Dockerfile", True),
        ("tpu", "docker/Dockerfile", False),
        ("tpu", "docker/tpu_patch/fix.patch", True),
        ("cu13", "docker/tpu_patch/fix.patch", False),
        (VARIANT, "source.py", False),
    ],
)
def test_pr_eligibility_uses_common_and_hardware_inputs(repository, variant, path, eligible):
    root, git = repository
    (root / path).write_text("changed\n")
    git("commit", "-qam", "PR merge")
    base = IMAGE_INPUTS.compute("HEAD^1", root=root, variant=variant)
    current = IMAGE_INPUTS.compute("HEAD", root=root, variant=variant)
    assert (current != base) is eligible
    if variant == "tpu":
        workflow_file, job, step_id, output_key = "pr-test-tpu.yml", "plan", "image", "build"
    else:
        workflow_file, job, step_id, output_key = "_build-pr-ci-image.yml", "docker-decide", "compare", "eligible"
    workflow = yaml.safe_load((ROOT / ".github/workflows" / workflow_file).read_text())
    step = next(step for step in workflow["jobs"][job]["steps"] if step.get("id") == step_id)
    # Execute the real reader and workflow against the committed fixture inputs.
    (root / "docker/image_inputs.py").write_text((ROOT / "docker/image_inputs.py").read_text())
    output = root / "output"
    subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=root,
        check=True,
        env={
            **os.environ,
            "ARCH": {"cu13": "amd64", VARIANT: "rocm10-mi35x"}.get(variant, ""),
            "BASE_SHA": "older-event-base",
            "PR_NUMBER": "42",
            "GITHUB_REPOSITORY": "radixark/miles",
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert f"{output_key}={str(eligible).lower()}" in output.read_text()


@pytest.mark.parametrize(
    "github_repository,path,variants",
    [
        ("radixark/miles", "requirements.txt", ["cu13", "tpu"]),
        ("radixark/miles", "docker/patch/cuda.patch", ["cu13"]),
        ("radixark/miles", "docker/tpu_patch/fix.patch", ["tpu"]),
        ("radixark/miles", "docker/Dockerfile.rocm", []),
        ("radixark/miles", "source.py", []),
        # The TPU image belongs to radixark/miles: a copy builds only cu13.
        ("example/miles-copy", "requirements.txt", ["cu13"]),
        ("example/miles-copy", "docker/tpu_patch/fix.patch", []),
    ],
)
def test_rolling_push_builds_only_supported_variants_with_changed_inputs(
    repository, github_repository, path, variants
):
    root, git = repository
    before = git("rev-parse", "HEAD")
    (root / path).write_text("changed\n")
    git("commit", "-qam", "push")
    git("remote", "add", "origin", str(root))
    (root / "docker/image_inputs.py").write_text((ROOT / "docker/image_inputs.py").read_text())
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker-build.yml").read_text())
    step = next(step for step in workflow["jobs"]["push-variants"]["steps"] if step.get("id") == "pick")
    output = root / "output"
    subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=root,
        check=True,
        env={**os.environ, "BEFORE": before, "GITHUB_REPOSITORY": github_repository, "GITHUB_OUTPUT": str(output)},
    )
    assert json.loads(output.read_text().removeprefix("variants=")) == variants
    assert "needs.push-variants.outputs.variants != '[]'" in workflow["jobs"]["build-and-push"]["if"]


def test_added_deleted_and_renamed_rocm_inputs_change_the_hash(repository):
    root, git = repository
    base = IMAGE_INPUTS.compute("HEAD", root=root, variant=VARIANT)
    git("mv", "docker/amd/install.py", "docker/amd/renamed.py")
    git("rm", "docker/amd_patch/latest/fix.patch")
    (root / "docker/amd_patch").mkdir(parents=True, exist_ok=True)
    (root / "docker/amd_patch/new.patch").write_text("new\n")
    git("add", ".")
    git("commit", "-qm", "changed inputs")
    assert IMAGE_INPUTS.compute("HEAD", root=root, variant=VARIANT) != base
    assert IMAGE_INPUTS.compute(root=root, variant=VARIANT) == IMAGE_INPUTS.compute("HEAD", root=root, variant=VARIANT)


@pytest.mark.parametrize("available", ["true", "false", ""])
@pytest.mark.parametrize("input_tag", ["", "frozen-rocm", "pr-42-rocm10-mi35x"])
@pytest.mark.parametrize("body", ["", "ci-image-tag: cuda-only\r\nci-rocm-image-tag: body-rocm\r\n"])
def test_image_precedence(tmp_path, monkeypatch, available, input_tag, body):
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_OUTPUT": str(output),
        "INPUT_CI_IMAGE_TAG": input_tag,
        "PR_IMAGE_AVAILABLE": available,
        "PR_NUMBER": "123",
        "PR_BODY": body,
    }.items():
        monkeypatch.setenv(key, value)
    assert rocm.main() == 0
    expected = (
        input_tag
        or ("pr-123-rocm10-mi35x" if available == "true" else "")
        or ("body-rocm" if body else rocm.DEFAULT_IMAGE_TAG)
    )
    assert output.read_text() == f"container_image={rocm.container_image(expected)}\n"


@pytest.mark.parametrize("tag", ["registry/image:tag", "bad/tag", "x\ny", "-tag", "a" * 129])
def test_invalid_tags_are_rejected(tag):
    with pytest.raises(ValueError):
        rocm.container_image(tag)


@pytest.mark.parametrize(
    "published,rebuild,needs_image", [("", True, True), ("old", True, True), ("current", False, True)]
)
def test_registry_decision_uses_the_rocm_tag(tmp_path, published, rebuild, needs_image):
    workflow = yaml.safe_load((ROOT / ".github/workflows/_build-pr-ci-image.yml").read_text())
    step = next(step for step in workflow["jobs"]["docker-decide"]["steps"] if step.get("id") == "decide")
    binary = tmp_path / "docker"
    binary.write_text(
        "#!/bin/sh\n"
        + f"printf '%s\\n' '{json.dumps({'config': {'Labels': {'miles.image-inputs': published}}})}'\n"
        + 'printf "%s\\n" "$*" > "$CALLS"\n'
    )
    binary.chmod(0o755)
    output, calls = tmp_path / "output", tmp_path / "calls"
    subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=ROOT,
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CURRENT": "current",
            "FORCE_REBUILD": "false",
            "INSPECT_TAG": "true",
            "PR_NUMBER": "123",
            "ARCH": "rocm10-mi35x",
            "GITHUB_REPOSITORY": "example/miles-copy",
            "GITHUB_OUTPUT": str(output),
            "CALLS": str(calls),
        },
    )
    assert output.read_text() == f"needs_image={str(needs_image).lower()}\nrebuild={str(rebuild).lower()}\n"
    assert "radixark/miles:pr-123-rocm10-mi35x" in calls.read_text()


@pytest.mark.parametrize("github_repository", ["radixark/miles", "example/miles-copy"])
@pytest.mark.parametrize("fork", [True, False])
def test_close_cleanup_deletes_only_the_expected_tags(tmp_path, fork, github_repository):
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker-pr-tag-cleanup.yml").read_text())
    script = workflow["jobs"]["delete-pr-tag"]["steps"][0]["run"]
    script = script[script.index("# One tag per PR_IMAGE_ARCHES") :].replace(
        "${{ github.event.pull_request.number }}", "123"
    )
    binary = tmp_path / "curl"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    binary.chmod(0o755)
    calls = tmp_path / "calls"
    subprocess.run(
        ["bash", "-c", script],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "FORK": str(fork).lower(),
            "TOKEN": "test",
            "GITHUB_REPOSITORY": github_repository,
            "CALLS": str(calls),
        },
    )
    deleted = calls.read_text().splitlines() if calls.exists() else []
    assert all(f"/repositories/{github_repository}/tags/" in line for line in deleted)
    tags = [line.rsplit("/", 2)[-2] for line in deleted]
    if github_repository == "radixark/miles":
        assert tags == (
            ["pr-123-rocm10-mi35x"] if fork else ["pr-123", "pr-123-arm64", "pr-123-rocm10-mi35x", "pr-123-tpu"]
        )
    else:
        # A copy never builds the ROCm or TPU PR images, so it leaves radixark/miles' tags alone.
        assert tags == ([] if fork else ["pr-123", "pr-123-arm64"])
