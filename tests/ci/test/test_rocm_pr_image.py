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

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

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
        "docker/Dockerfile",
        "docker/Dockerfile.rocm",
        "docker/Dockerfile.tpu",
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
    "path,eligible",
    [
        ("docker/Dockerfile.rocm", True),
        ("docker/amd_patch/latest/fix.patch", True),
        ("docker/amd/install.py", True),
        ("requirements.txt", True),
        ("docker/Dockerfile", False),
        ("docker/patch/cuda.patch", False),
        ("docker/Dockerfile.tpu", False),
        ("source.py", False),
    ],
)
def test_rocm_eligibility_uses_the_merge_base(repository, path, eligible):
    root, git = repository
    (root / path).write_text("changed\n")
    git("commit", "-qam", "PR merge")
    workflow = yaml.safe_load((ROOT / ".github/workflows/_build-pr-ci-image.yml").read_text())
    step = next(step for step in workflow["jobs"]["docker-decide"]["steps"] if step.get("id") == "compare")
    # Use the actual workflow's comparison and a trusted local hash reader.
    reader = root / "hash.py"
    # Keep the real module, overriding only its source root at its documented API.
    base = IMAGE_INPUTS.compute("HEAD^1", root=root, variant=VARIANT)
    current = IMAGE_INPUTS.compute("HEAD", root=root, variant=VARIANT)
    assert (current != base) is eligible
    # Run the shell decision with a tiny adapter forwarding --rev/--variant.
    reader.write_text(
        "import argparse, runpy\nfrom pathlib import Path\n"
        f"d=runpy.run_path({str(ROOT / 'docker/image_inputs.py')!r})\n"
        "p=argparse.ArgumentParser(); p.add_argument('--rev'); p.add_argument('--variant'); a=p.parse_args()\n"
        f"print(d['compute'](a.rev,root=Path({str(root)!r}),variant=a.variant))\n"
    )
    code = step["run"].replace("python3 docker/image_inputs.py", f"{sys.executable} {reader}")
    output = root / "output"
    subprocess.run(
        ["bash", "-c", code],
        cwd=root,
        check=True,
        env={**os.environ, "IMAGE_VARIANT": VARIANT, "BASE_SHA": "older-event-base", "GITHUB_OUTPUT": str(output)},
    )
    assert f"eligible={str(eligible).lower()}" in output.read_text()


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
            "TAG_SUFFIX": "-rocm10-mi35x",
            "GITHUB_OUTPUT": str(output),
            "CALLS": str(calls),
        },
    )
    assert output.read_text() == f"needs_image={str(needs_image).lower()}\nrebuild={str(rebuild).lower()}\n"
    assert "radixark/miles:pr-123-rocm10-mi35x" in calls.read_text()


@pytest.mark.parametrize("fork", [True, False])
def test_close_cleanup_deletes_only_the_expected_tags(tmp_path, fork):
    workflow = yaml.safe_load((ROOT / ".github/workflows/docker-pr-tag-cleanup.yml").read_text())
    script = workflow["jobs"]["delete-pr-tag"]["steps"][0]["run"]
    script = script[script.index("# CUDA, TPU") :].replace("${{ github.event.pull_request.number }}", "123")
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
            "CALLS": str(calls),
        },
    )
    tags = [line.rsplit("/", 2)[-2] for line in calls.read_text().splitlines()]
    assert tags == (["pr-123-rocm10-mi35x"] if fork else ["pr-123", "pr-123-tpu", "pr-123-rocm10-mi35x"])
