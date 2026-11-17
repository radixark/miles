#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Validate a fork image request for default-branch build orchestration."""

import argparse
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))

import image_inputs  # noqa: E402
from tests.ci.ci_policy import resolve_workflow_inputs  # noqa: E402

API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
CPU_JOBS = {f"stage-a-cpu ({partition}) / run-cpu" for partition in range(4)}
REQUEST_KEYS = {"pr", "merge_sha", "inputs_hash", "run_id", "run_attempt", "force_rebuild", "labels"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def open_api(url, method="GET"):
    # Self-hosted builders have no gh CLI. Artifact downloads redirect to blob storage, which must not get the token.
    request = urllib.request.Request(url, method=method, headers={"Accept": "application/vnd.github+json"})
    request.add_unredirected_header("Authorization", f"Bearer {os.environ['GH_TOKEN']}")
    return urllib.request.urlopen(request, timeout=60)


def api(path, *, method="GET", collection=None):
    url = f"{API_URL}/{path}"
    items = []
    while True:
        with open_api(url, method) as response:
            data = response.read()
            next_page = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
        result = json.loads(data) if data.strip() else None
        if not collection:
            return result
        items += result[collection]
        if not next_page:
            return items
        url = next_page.group(1)


def parse_request(data):
    require(len(data) <= 65536, "Request archive is too large")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(archive.namelist() == ["request.json"], "Expected only request.json")
        info = archive.getinfo("request.json")
        require(info.file_size <= 4096, "Request JSON is too large")
        request = json.loads(archive.read(info))
    require(isinstance(request, dict) and request.keys() == REQUEST_KEYS, "Invalid request fields")
    for key in ("pr", "run_id", "run_attempt"):
        require(type(request[key]) is int and request[key] > 0, f"Invalid {key}")
    require(type(request["force_rebuild"]) is bool, "Invalid force_rebuild")
    require(
        isinstance(request["labels"], list) and all(isinstance(label, str) for label in request["labels"]),
        "Invalid labels",
    )
    for key, length in (("merge_sha", 40), ("inputs_hash", 64)):
        require(
            isinstance(request[key], str) and re.fullmatch(f"[0-9a-f]{{{length}}}", request[key]), f"Invalid {key}"
        )
    return request


def validate_identity(request, run, pr, repository, workflow_id):
    require(run["id"] == request["run_id"] and run["run_attempt"] == request["run_attempt"], "Stale run attempt")
    require(run["event"] == "pull_request" and run["status"] == "in_progress", "Expected an active PR run")
    require(run["repository"]["full_name"] == repository, "Wrong run repository")
    require(
        run["workflow_id"] == workflow_id and run["path"].split("@")[0] == ".github/workflows/pr-test.yml",
        "Wrong source workflow",
    )
    require(pr["number"] == request["pr"] and pr["state"] == "open", "PR is no longer open")
    require(pr["base"]["repo"]["full_name"] == repository, "Wrong PR base repository")
    head = pr["head"]
    require(head["repo"] and head["repo"]["id"] != pr["base"]["repo"]["id"], "Expected a fork PR")
    require(head["repo"]["id"] == run["head_repository"]["id"], "Wrong fork repository")
    require(head["sha"] == run["head_sha"] and head["ref"] == run["head_branch"], "PR head moved")
    # Reruns keep the original merge SHA even when main has advanced.
    reference = f"{repository}/.github/workflows/_build-pr-ci-image.yml@{request['merge_sha']}"
    require(
        any(
            entry["path"] == reference
            and entry.get("ref") == f"refs/pull/{request['pr']}/merge"
            and entry["sha"] == request["merge_sha"]
            for entry in run.get("referenced_workflows", [])
        ),
        "Merge SHA is not the image workflow's frozen source",
    )


def current_request(request, repository):
    run = api(f"repos/{repository}/actions/runs/{request['run_id']}")
    pr = api(f"repos/{repository}/pulls/{request['pr']}")
    workflow = api(f"repos/{repository}/actions/workflows/pr-test.yml")
    validate_identity(request, run, pr, repository, workflow["id"])
    head = pr["head"]
    query = urlencode({"event": "pull_request", "head_sha": head["sha"], "per_page": 100})
    runs = api(f"repos/{repository}/actions/workflows/pr-test.yml/runs?{query}", collection="workflow_runs")
    # Simultaneous label events start several runs; the concurrency group may cancel a higher-numbered one.
    matching = [
        item["id"]
        for item in runs
        if item["head_repository"]["id"] == head["repo"]["id"]
        and item["head_branch"] == head["ref"]
        and item["conclusion"] != "cancelled"
    ]
    require(matching and max(matching) == run["id"], "A newer PR Test run superseded this request")
    # Match the caller's event snapshot, including on reruns and after label removal.
    policy = resolve_workflow_inputs("pull_request", "", json.dumps(request["labels"]))
    if not policy.bypass_fastfail:
        jobs = api(
            f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=all&per_page=100",
            collection="jobs",
        )
        # Partial reruns retain successful dependencies from earlier attempts.
        cpu = {}
        for job in sorted(jobs, key=lambda job: job["run_attempt"]):
            if job["name"] in CPU_JOBS and job["run_attempt"] <= request["run_attempt"]:
                cpu[job["name"]] = job
        require(
            cpu.keys() == CPU_JOBS and all(job["conclusion"] == "success" for job in cpu.values()),
            "CPU A gate did not pass",
        )
    return pr


def resolve_request(event, repository):
    run = event["workflow_run"]
    artifacts = api(f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100", collection="artifacts")
    name = f"fork-ci-image-{run['id']}-{run['run_attempt']}"
    matches = [item for item in artifacts if item["name"] == name and not item["expired"]]
    if not matches:
        return None
    require(len(matches) == 1 and matches[0]["size_in_bytes"] <= 65536, "Invalid request artifact")
    with open_api(f"{API_URL}/repos/{repository}/actions/artifacts/{matches[0]['id']}/zip") as response:
        request = parse_request(response.read())
    require(
        request["run_id"] == run["id"] and request["run_attempt"] == run["run_attempt"],
        "Request belongs to another attempt",
    )
    current_request(request, repository)
    return request


def find_publisher(source, repository):
    # `workflow_run.in_progress` can fire several times per attempt; only the earliest triggered run publishes.
    workflow = api(f"repos/{repository}/actions/workflows/build-fork-ci-image.yml")
    query = urlencode({"event": "workflow_run", "created": f">={source['created_at']}", "per_page": 100})
    title = f"fork-ci-image-{source['id']}-{source['run_attempt']}"
    runs = api(f"repos/{repository}/actions/workflows/{workflow['id']}/runs?{query}", collection="workflow_runs")
    matches = [
        run
        for run in runs
        if run["display_title"] == title and run["event"] == "workflow_run" and run["workflow_id"] == workflow["id"]
    ]
    return min(matches, key=lambda run: run["id"], default=None)


def wait_request(event, repository, run_id):
    source = event["workflow_run"]
    publisher = find_publisher(source, repository)
    if publisher and publisher["id"] < run_id:
        print(f"{publisher['html_url']} publishes this attempt", flush=True)
        return None
    while True:
        run = api(f"repos/{repository}/actions/runs/{source['id']}")
        if run["run_attempt"] != source["run_attempt"] or run["status"] == "completed":
            return None
        request = resolve_request(event, repository)
        if request:
            return request
        jobs = api(
            f"repos/{repository}/actions/runs/{source['id']}/jobs?filter=latest&per_page=100", collection="jobs"
        )
        if any(
            job["name"] in ("docker-build-amd64", "docker-build-amd64 / docker-build")
            and job["run_attempt"] == source["run_attempt"]
            and job["status"] == "completed"
            for job in jobs
        ):
            return None
        time.sleep(30)


def require_active_source(request, repository):
    run = api(f"repos/{repository}/actions/runs/{request['run_id']}")
    require(
        run["run_attempt"] == request["run_attempt"] and run["status"] == "in_progress",
        "Source CI attempt is no longer active",
    )
    return run


def wait_build(request, repository, arch):
    # Wait for this architecture's job only: amd64 gates the GPU suites, arm64 must not delay them.
    job_name = f"build ({arch})"
    publisher = None
    while True:
        source = require_active_source(request, repository)
        if publisher is None:
            publisher = find_publisher(source, repository)
            if publisher:
                print(f"Waiting for {job_name} in {publisher['html_url']}", flush=True)
        else:
            publisher = api(f"repos/{repository}/actions/runs/{publisher['id']}")
        if publisher:
            jobs = api(
                f"repos/{repository}/actions/runs/{publisher['id']}/jobs?filter=latest&per_page=100", collection="jobs"
            )
            job = next((job for job in jobs if job["name"] == job_name), None)
            if job and job["status"] == "completed":
                require(job["conclusion"] == "success", f"{job_name} {job['conclusion']}: {publisher['html_url']}")
                return
            require(
                publisher["status"] != "completed",
                f"Image publication {publisher['conclusion']} without {job_name}: {publisher['html_url']}",
            )
        time.sleep(30)


def build_image(request, repository, source, arch):
    require_active_source(request, repository)
    variant, tag = image_inputs.pr_image(arch, request["pr"])
    command = [
        sys.executable,
        str(ROOT / "docker/build.py"),
        "--variant",
        variant,
        "--image-tag",
        "custom",
        "--custom-tag",
        tag,
        "--context",
        str(source),
        "--output",
        f"type=oci,dest={os.environ['OCI_OUTPUT']},tar=false",
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                require_active_source(request, repository)
                continue
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # build.py may exit before a buildx child that ignores SIGTERM.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                process.wait()


def validate_source(request, pr, source):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

    require(git("rev-parse", "HEAD") == request["merge_sha"], "Checkout is not the requested merge")
    parents = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    require(len(parents) == 3 and parents[2] == pr["head"]["sha"], "Merge does not contain the fork head")
    # No host-side reader may follow a PR symlink into the trusted checkout or credentials.
    paths = image_inputs._paths_at("HEAD", source) + [
        "docker/Dockerfile",
        ".dockerignore",
        "docker/Dockerfile.dockerignore",
    ]
    for name in paths:
        path = source
        for component in Path(name).parts:
            path /= component
            require(not path.is_symlink(), f"Build input must not be a symlink: {name}")
    current = image_inputs.compute("HEAD", root=source)
    require(current == request["inputs_hash"], "Requested inputs hash does not match trusted computation")
    require(current != image_inputs.compute("HEAD^1", root=source), "Build inputs match the base")
    require(current == image_inputs.compute(root=source), "Build context differs from the frozen inputs")


def verify_published(request, arch):
    _, tag = image_inputs.pr_image(arch, request["pr"])
    published = subprocess.check_output(
        ["docker", "buildx", "imagetools", "inspect", f"radixark/miles:{tag}", "--format", "{{ json .Image }}"],
        text=True,
    )
    require(image_inputs.read_label(published) == request["inputs_hash"], f"Published {tag} inputs do not match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["resolve", "wait-build", "check-source", "build", "check-current", "finish"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--arch", choices=sorted(image_inputs.PR_IMAGE_ARCHES))
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "wait-build":
        wait_build(
            dict(run_id=int(os.environ["GITHUB_RUN_ID"]), run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"])),
            repository,
            args.arch,
        )
        return
    if args.mode == "resolve":
        request = wait_request(
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), repository, int(os.environ["GITHUB_RUN_ID"])
        )
        if request:
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(
                    f"request={json.dumps(request, separators=(',', ':'))}\npr={request['pr']}\nmerge_sha={request['merge_sha']}\n"
                )
        return

    request = json.loads(os.environ["REQUEST_JSON"])
    pr = current_request(request, repository)
    validate_source(request, pr, args.source.resolve())
    if args.mode == "build":
        build_image(request, repository, args.source.resolve(), args.arch)
    elif args.mode == "finish":
        verify_published(request, args.arch)
        # The amd64 build owns the one-shot label, as on the same-repository path.
        if request["force_rebuild"] and args.arch == "amd64":
            try:
                api(f"repos/{repository}/issues/{request['pr']}/labels/rebuild-ci-image", method="DELETE")
            except OSError:
                print(
                    "::warning::Could not remove the rebuild-ci-image label; remove it by hand or every run will rebuild."
                )


if __name__ == "__main__":
    main()
