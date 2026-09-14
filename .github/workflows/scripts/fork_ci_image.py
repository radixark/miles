#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Validate a fork image request for default-branch build orchestration."""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))

import image_inputs  # noqa: E402
from tests.ci.ci_policy import resolve_policy, resolve_workflow_inputs  # noqa: E402
from tests.ci.file_run import collect_snapshot_tests  # noqa: E402
from tests.ci.hardware import CUDA_STAGES  # noqa: E402
from tests.ci.stage_selection import select_skipped_gpu_stages  # noqa: E402

CPU_JOBS = {f"stage-a-cpu ({partition}) / run-cpu" for partition in range(4)}
REQUEST_KEYS = {"pr", "merge_sha", "inputs_hash", "run_id", "run_attempt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def api(path, *, method="GET", collection=None):
    command = ["gh", "api", "-X", method, path]
    if collection:
        command += ["--paginate", "--slurp"]
    data = subprocess.check_output(command)
    result = json.loads(data) if data.strip() else None
    if collection:
        return [item for page in result for item in page[collection]]
    return result


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
    for key, length in (("merge_sha", 40), ("inputs_hash", 64)):
        require(
            isinstance(request[key], str) and re.fullmatch(f"[0-9a-f]{{{length}}}", request[key]), f"Invalid {key}"
        )
    return request


def validate_identity(request, run, pr, repository, workflow_id):
    require(run["id"] == request["run_id"] and run["run_attempt"] == request["run_attempt"], "Stale run attempt")
    require(run["event"] == "pull_request" and run["status"] == "completed", "Expected a completed PR run")
    require(run["conclusion"] == "failure", "Expected the failed image handoff")
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
    matching = [
        item["id"]
        for item in runs
        if item["head_repository"]["id"] == head["repo"]["id"] and item["head_branch"] == head["ref"]
    ]
    require(matching and max(matching) == run["id"], "A newer PR Test run superseded this request")
    labels = [label["name"] for label in pr["labels"]]
    policy = resolve_workflow_inputs("pull_request", "", json.dumps(labels))
    if not policy.bypass_fastfail:
        jobs = api(
            f"repos/{repository}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100",
            collection="jobs",
        )
        cpu = [job for job in jobs if job["name"] in CPU_JOBS]
        require(
            len(cpu) == len(CPU_JOBS)
            and {job["name"] for job in cpu} == CPU_JOBS
            and all(job["conclusion"] == "success" for job in cpu),
            "CPU A gate did not pass in this attempt",
        )
    return pr, policy


def resolve_request(event, repository):
    run = event["workflow_run"]
    artifacts = api(f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100", collection="artifacts")
    name = f"fork-ci-image-{run['id']}-{run['run_attempt']}"
    matches = [item for item in artifacts if item["name"] == name and not item["expired"]]
    if not matches:
        return None
    require(len(matches) == 1 and matches[0]["size_in_bytes"] <= 65536, "Invalid request artifact")
    data = subprocess.check_output(["gh", "api", f"repos/{repository}/actions/artifacts/{matches[0]['id']}/zip"])
    request = parse_request(data)
    require(
        request["run_id"] == run["id"] and request["run_attempt"] == run["run_attempt"],
        "Request belongs to another attempt",
    )
    current_request(request, repository)
    return request


def validate_source(request, pr, policy, source):
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
    run_policy = resolve_policy(policy.cadence, set(policy.raw_labels))
    skipped = select_skipped_gpu_stages(
        event_name="pull_request",
        changed_files=None,
        registrations=collect_snapshot_tests(source),
        run_policy=run_policy,
        raw_labels=policy.raw_labels,
    )
    require(CUDA_STAGES.keys() - set(skipped), "No CUDA tests selected")


def verify_published(request):
    manifest = json.loads(image_inputs.inspect_published(f"radixark/miles:pr-{request['pr']}"))
    for platform in ("linux/amd64", "linux/arm64"):
        configs = [config for key, config in manifest.items() if key == platform or key.startswith(platform + "/")]
        require(
            configs
            and all(image_inputs.read_label(json.dumps(config)) == request["inputs_hash"] for config in configs),
            f"Published {platform} inputs do not match",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["resolve", "check-source", "check-current", "finish"])
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "resolve":
        request = resolve_request(json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), repository)
        if request:
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(
                    f"request={json.dumps(request, separators=(',', ':'))}\npr={request['pr']}\nmerge_sha={request['merge_sha']}\n"
                )
        return

    request = json.loads(os.environ["REQUEST_JSON"])
    pr, policy = current_request(request, repository)
    validate_source(request, pr, policy, args.source.resolve())
    image = f"radixark/miles:pr-{request['pr']}"
    if args.mode == "check-source":
        if "rebuild-ci-image" not in [label["name"] for label in pr["labels"]]:
            published = image_inputs.read_label(image_inputs.inspect_published(image))
            require(published != request["inputs_hash"], "Image already matches; refusing a repeated build request")
    elif args.mode == "finish":
        verify_published(request)
        if "rebuild-ci-image" in [label["name"] for label in pr["labels"]]:
            api(f"repos/{repository}/issues/{request['pr']}/labels/rebuild-ci-image", method="DELETE")
        current_request(request, repository)
        # Rerunning only failed jobs would retain docker-decide's rebuild=true output.
        api(f"repos/{repository}/actions/runs/{request['run_id']}/rerun", method="POST")


if __name__ == "__main__":
    main()
