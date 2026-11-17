#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Validate image requests and orchestrate builds with default-branch code.

CUDA forks use docker-build; all ROCm PR builds use Spur compute nodes.

Fork driver code is never executed. Literal variant build_args edits are accepted
against the frozen merge base; ROCm builds use only rocm10-mi35x arguments.
"""

import argparse
import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker"))

import image_inputs  # noqa: E402
from tests.ci.ci_policy import resolve_policy, resolve_workflow_inputs  # noqa: E402
from tests.ci.file_run import collect_snapshot_tests  # noqa: E402
from tests.ci.rocm import ROCM_STAGES  # noqa: E402
from tests.ci.rocm_file_plan import is_opted_in, plan_jobs  # noqa: E402
from tests.ci.stage_selection import read_changed_files, select_skipped_gpu_stages  # noqa: E402

CPU_JOBS = {f"stage-a-cpu ({partition}) / run-cpu" for partition in range(4)}
REQUEST_KEYS = {"pr", "merge_sha", "inputs_hash", "run_id", "run_attempt", "force_rebuild", "labels"}
SOURCE_WORKFLOWS = {"cu13": "pr-test.yml", "rocm10-mi35x-pr": "pr-test-rocm.yml"}


def variant(request):
    # Older CUDA requests retain their original schema and meaning.
    return request.get("variant", "cu13")


def image_tag(request):
    suffix = "-rocm10-mi35x" if variant(request) == "rocm10-mi35x-pr" else ""
    return f"pr-{request['pr']}{suffix}"


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
    require(
        isinstance(request, dict) and request.keys() in (REQUEST_KEYS, REQUEST_KEYS | {"variant"}),
        "Invalid request fields",
    )
    require(isinstance(variant(request), str) and variant(request) in SOURCE_WORKFLOWS, "Invalid variant")
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
        run["workflow_id"] == workflow_id
        and run["path"].split("@")[0] == f".github/workflows/{SOURCE_WORKFLOWS[variant(request)]}",
        "Wrong source workflow",
    )
    require(pr["number"] == request["pr"] and pr["state"] == "open", "PR is no longer open")
    require(pr["base"]["repo"]["full_name"] == repository, "Wrong PR base repository")
    head = pr["head"]
    require(head["repo"], "PR head repository was deleted")
    if variant(request) == "cu13":
        require(head["repo"]["id"] != pr["base"]["repo"]["id"], "Expected a fork PR")
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
    filename = SOURCE_WORKFLOWS[variant(request)]
    workflow = api(f"repos/{repository}/actions/workflows/{filename}")
    validate_identity(request, run, pr, repository, workflow["id"])
    head = pr["head"]
    query = urlencode({"event": "pull_request", "head_sha": head["sha"], "per_page": 100})
    runs = api(f"repos/{repository}/actions/workflows/{filename}/runs?{query}", collection="workflow_runs")
    matching = [
        item["id"]
        for item in runs
        if item["head_repository"]["id"] == head["repo"]["id"] and item["head_branch"] == head["ref"]
    ]
    require(matching and max(matching) == run["id"], "A newer PR Test run superseded this request")
    # Match the caller's event snapshot, including on reruns and after label removal.
    policy = resolve_workflow_inputs("pull_request", "", json.dumps(request["labels"]))
    # ROCm's caller has no CPU A gate. Preserve CUDA's existing success/bypass
    # check; both callers still validate the frozen build context.
    if variant(request) == "cu13" and not policy.bypass_fastfail:
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
    data = subprocess.check_output(["gh", "api", f"repos/{repository}/actions/artifacts/{matches[0]['id']}/zip"])
    request = parse_request(data)
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
            job["name"] in ("docker-build", "docker-build / docker-build")
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


def wait_build(request, repository):
    publisher = None
    while True:
        source = require_active_source(request, repository)
        if publisher is None:
            publisher = find_publisher(source, repository)
            if publisher:
                print(f"Waiting for {publisher['html_url']}", flush=True)
        else:
            publisher = api(f"repos/{repository}/actions/runs/{publisher['id']}")
        if publisher and publisher["status"] == "completed":
            require(
                publisher["conclusion"] == "success",
                f"Image publication {publisher['conclusion']}: {publisher['html_url']}",
            )
            jobs = api(
                f"repos/{repository}/actions/runs/{publisher['id']}/jobs?filter=latest&per_page=100", collection="jobs"
            )
            require(
                any(job["name"] == "build" and job["conclusion"] == "success" for job in jobs),
                "Publisher completed without a successful image build",
            )
            return
        time.sleep(30)


def _rocm_recipe(text):
    """Return the ROCm PR arguments and mask all literal variant argument fields."""

    def field(node, name):
        require(isinstance(node, ast.Dict), "Expected a literal recipe dictionary")
        matches = [
            (index, value)
            for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True))
            if isinstance(key, ast.Constant) and key.value == name
        ]
        require(len(matches) == 1, f"Expected one {name} field")
        return matches[0]

    try:
        tree = ast.parse(text)
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets)
        ]
        require(len(values) == 1, "Expected one literal VARIANTS assignment")
        _, recipe = field(values[0], "rocm10-mi35x")
        _, arguments_node = field(recipe, "build_args")
        arguments = ast.literal_eval(arguments_node)
    except (SyntaxError, ValueError, TypeError, KeyError) as error:
        raise ValueError("Fork ROCm images require literal rocm10-mi35x build_args in docker/build.py") from error
    require(
        isinstance(arguments, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in arguments.items()),
        "Fork ROCm build_args must be a dictionary of strings",
    )
    for entry in values[0].values:
        if isinstance(entry, ast.Dict):
            for index, key in enumerate(entry.keys):
                if isinstance(key, ast.Constant) and key.value == "build_args":
                    entry.values[index] = ast.Dict(keys=[], values=[])
    return arguments, ast.dump(tree)


def rocm_build_args(source):
    # Treat the fork's driver as data, never as host-side Python.
    try:
        text = (source / "docker/build.py").read_text()
    except OSError as error:
        raise ValueError("Fork ROCm images require docker/build.py") from error
    return _rocm_recipe(text)[0]


def build_command(request, source):
    command = [
        sys.executable,
        str(ROOT / "docker/build.py"),
        "--variant",
        variant(request),
        "--image-tag",
        "custom",
        "--custom-tag",
        f"pr-{request['pr']}",
        "--context",
        str(source),
        "--output",
        f"type=oci,dest={os.environ['OCI_OUTPUT']},tar=false",
    ]
    if variant(request) == "rocm10-mi35x-pr":
        arguments = rocm_build_args(source)
        # CLI overrides cannot remove a trusted default. Refuse that unsupported
        # edit instead of silently building a recipe different from the fork's.
        require(rocm_build_args(ROOT).keys() <= arguments.keys(), "Fork ROCm build_args cannot remove driver defaults")
        for key, value in arguments.items():
            command += ["--build-arg", f"{key}={value}"]
    return command


def build_image(request, repository, source):
    require_active_source(request, repository)
    command = build_command(request, source)
    if variant(request) == "rocm10-mi35x-pr":
        command = [
            sys.executable,
            str(ROOT / ".github/workflows/scripts/spur/build_image.py"),
            "launch",
            "--source",
            str(source),
            "--merge-sha",
            request["merge_sha"],
            "--inputs-hash",
            request["inputs_hash"],
            "--command",
            json.dumps(command),
        ]
    build_env = dict(os.environ)
    # Newer main supports forwarding this as a BuildKit secret. PR Dockerfiles
    # must remain anonymous even if the login runner happens to define it.
    build_env.pop("BUILD_GITHUB_TOKEN", None)
    process = subprocess.Popen(command, start_new_session=True, env=build_env)
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
    recipe = "docker/Dockerfile.rocm" if variant(request) == "rocm10-mi35x-pr" else "docker/Dockerfile"
    paths = image_inputs._paths_at("HEAD", source, variant(request)) + [
        recipe,
        ".dockerignore",
        f"{recipe}.dockerignore",
    ]
    for name in paths:
        path = source
        for component in Path(name).parts:
            path /= component
            require(not path.is_symlink(), f"Build input must not be a symlink: {name}")
    current = image_inputs.compute("HEAD", root=source, variant=variant(request))
    require(current == request["inputs_hash"], "Requested inputs hash does not match trusted computation")
    require(
        current != image_inputs.compute("HEAD^1", root=source, variant=variant(request)), "Build inputs match the base"
    )
    require(
        current == image_inputs.compute(root=source, variant=variant(request)),
        "Build context differs from the frozen inputs",
    )
    if variant(request) == "rocm10-mi35x-pr":
        base_driver = subprocess.check_output(["git", "show", "HEAD^1:docker/build.py"], cwd=source, text=True)
        driver = (source / "docker/build.py").read_text()
        if driver != base_driver:
            require(
                _rocm_recipe(driver)[1] == _rocm_recipe(base_driver)[1],
                "Fork ROCm images only honor literal build_args edits in docker/build.py",
            )
        policy = resolve_workflow_inputs("pull_request", "", json.dumps(request["labels"]))
        run_policy = resolve_policy(policy.cadence, set(policy.raw_labels))
        registrations = collect_snapshot_tests(source)
        skipped = select_skipped_gpu_stages(
            event_name="pull_request",
            changed_files=None,
            registrations=registrations,
            run_policy=run_policy,
            raw_labels=policy.raw_labels,
        )
        selected = ROCM_STAGES.keys() - set(skipped)
        jobs = []
        labels = set(policy.raw_labels)
        if not selected and is_opted_in(labels):
            # Parse the frozen diff with the same reader as the caller. Only
            # trusted code reads registrations; no fork Python is imported.
            diff = subprocess.check_output(["git", "diff", "--name-status", "-z", "-M", "HEAD^1", "HEAD"], cwd=source)
            with tempfile.NamedTemporaryFile() as changed:
                changed.write(diff)
                changed.flush()
                changes = read_changed_files(changed.name)
            if changes:
                jobs, _ = plan_jobs(changes, registrations, run_policy, labels, set(skipped))
        require(selected or jobs, "No ROCm tests selected")


def verify_published(request):
    manifest = json.loads(
        subprocess.check_output(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                f"radixark/miles:{image_tag(request)}",
                "--format",
                "{{ json .Image }}",
            ],
            text=True,
        )
    )
    if variant(request) == "rocm10-mi35x-pr":
        # Imagetools returns a config directly for a single-platform manifest,
        # and a platform map for an index (e.g. one with provenance attached).
        if manifest.get("architecture") == "amd64" and manifest.get("os") == "linux":
            require(
                image_inputs.read_label(json.dumps(manifest)) == request["inputs_hash"],
                "Published linux/amd64 inputs do not match",
            )
            return
        platforms = ("linux/amd64",)
    else:
        platforms = ("linux/amd64", "linux/arm64")
    for platform in platforms:
        configs = [config for key, config in manifest.items() if key == platform or key.startswith(platform + "/")]
        require(
            configs
            and all(image_inputs.read_label(json.dumps(config)) == request["inputs_hash"] for config in configs),
            f"Published {platform} inputs do not match",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["resolve", "wait-build", "check-source", "build", "check-current", "finish"])
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "wait-build":
        wait_build(
            dict(run_id=int(os.environ["GITHUB_RUN_ID"]), run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"])),
            repository,
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
                    f"image_tag={image_tag(request)}\nvariant={variant(request)}\n"
                )
        return

    request = json.loads(os.environ["REQUEST_JSON"])
    pr = current_request(request, repository)
    validate_source(request, pr, args.source.resolve())
    if args.mode == "build":
        build_image(request, repository, args.source.resolve())
    elif args.mode == "finish":
        verify_published(request)
        if request["force_rebuild"]:
            try:
                api(f"repos/{repository}/issues/{request['pr']}/labels/rebuild-ci-image", method="DELETE")
            except subprocess.CalledProcessError:
                print(
                    "::warning::Could not remove the rebuild-ci-image label; remove it by hand or every run will rebuild."
                )


if __name__ == "__main__":
    main()
