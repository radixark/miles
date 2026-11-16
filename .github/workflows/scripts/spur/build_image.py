#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
"""Build ROCm PR OCI layouts on Spur compute nodes with trusted orchestration.

The login runner submits a private manifest and streams the job log. Only a
sanitized environment reaches Slurm. The compute job uses its own scratch,
Docker config and BuildKit, removes them, then reports completion. Publication
credentials stay on the login runner; closed-PR tag deletion uses the existing
trusted hosted cleanup workflow, as TPU does.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Shared with _run-ci-rocm.yml: keep Miles jobs off other projects' nodes.
EXCLUDE = ",".join(json.loads(Path(__file__).with_name("other-project-nodes.json").read_text()))


def scheduler_env():
    # sbatch's default ALL would otherwise copy runner .env/repository secrets.
    # ALL with this small input avoids NONE's implicit login-environment load.
    return {"PATH": PATH, **{key: os.environ[key] for key in ("SLURM_CONF",) if key in os.environ}}


def queue_output(*arguments):
    # Like job.sh's job_alive: one empty controller response is not a terminal
    # state. Require two empty reads; command errors remain errors.
    for attempt in range(2):
        value = subprocess.check_output(["squeue", "--noheader", *arguments], env=scheduler_env(), text=True).strip()
        if value:
            return value
        if attempt == 0:
            time.sleep(5)
    return ""


def queued(job_id):
    return bool(queue_output("--jobs", job_id, "--format=%T"))


def cancel(run_dir):
    job_file = run_dir / "job-id"
    if not job_file.exists():
        # Cover cancellation after sbatch accepted the job but before its reply
        # was recorded. Recover the unique name with the same two-read rule.
        ids = queue_output("--name", f"miles-pr-image-{run_dir.name}", "--format=%i").split()
        for job_id in ids:
            stop_job(job_id)
        return
    stop_job(job_file.read_text().strip())


def stop_job(job_id):
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Invalid recorded Spur job ID")
    # Always cancel before consulting the queue, even after an empty response.
    # An already-completed ID can make scancel fail; accept it only after two
    # empty reads confirm absence. Other errors with a live job fail cleanup.
    result = subprocess.run(["scancel", job_id], env=scheduler_env())
    deadline = time.monotonic() + 120
    while queued(job_id):
        result.check_returncode()
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Spur job {job_id} did not stop; retaining its shared files")
        time.sleep(5)


def interrupted(signum, frame):
    raise RuntimeError(f"Spur build interrupted by signal {signum}")


def launch(source, merge_sha, inputs_hash, command):
    run_dir = Path(os.environ["OCI_OUTPUT"]).resolve().parent
    scratch = os.environ["CI_NODE_SCRATCH"]
    settings = {name: os.environ[name] for name in ("SLURM_PARTITION", "SLURM_ACCOUNT", "SLURM_QOS")}
    manifest = run_dir / "build.json"
    manifest.write_text(
        json.dumps(
            dict(
                source=str(source),
                merge_sha=merge_sha,
                inputs_hash=inputs_hash,
                command=command,
                scratch=scratch,
                trusted=str(ROOT),
            )
        )
    )
    manifest.chmod(0o600)
    job_script = run_dir / "job.sh"
    job_script.write_text(
        "#!/bin/bash\nexec /usr/bin/python3 "
        + shlex.join([str(Path(__file__).resolve()), "node", "--run-dir", str(run_dir)])
        + "\n"
    )
    job_script.chmod(0o700)
    submission = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        "--nodes=1",
        "--exclusive",
        "--mem=0",
        "--time=02:00:00",
        "--signal=B:TERM@120",
        "--partition",
        settings["SLURM_PARTITION"],
        "--account",
        settings["SLURM_ACCOUNT"],
        "--qos",
        settings["SLURM_QOS"],
        "--exclude",
        EXCLUDE,
        "--job-name",
        f"miles-pr-image-{run_dir.name}",
        "--output",
        str(run_dir / "build.log"),
        str(job_script),
    ]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        job_id = subprocess.check_output(submission, env=scheduler_env(), text=True).strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("sbatch did not return a numeric job ID")
        (run_dir / "job-id").write_text(job_id)
        print(f"Spur image build: job {job_id}", flush=True)
        wait_for_job(job_id, run_dir)
    finally:
        # A source-run cancellation also terminates the exact cluster job.
        # Do not let a second SIGTERM interrupt the cancellation confirmation.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cancel(run_dir)


def wait_for_job(job_id, run_dir):
    deadline = time.monotonic() + 150 * 60  # includes queueing, within the publisher's 180m budget
    offset = 0
    while queued(job_id):
        log = run_dir / "build.log"
        if log.exists():
            with log.open() as stream:
                stream.seek(offset)
                print(stream.read(), end="", flush=True)
                offset = stream.tell()
        if time.monotonic() >= deadline:
            raise TimeoutError("Spur image build exceeded its queue/build budget")
        time.sleep(5)
    # NFS can expose the exit marker later than Slurm's terminal state.
    for _ in range(18):
        if (run_dir / "exit-code").exists():
            break
        time.sleep(5)
    log = run_dir / "build.log"
    if log.exists():
        with log.open() as stream:
            stream.seek(offset)
            print(stream.read(), end="", flush=True)
    code = int((run_dir / "exit-code").read_text())
    if code:
        raise RuntimeError(f"Spur image build/cleanup failed ({code})")


def node(run_dir):
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    manifest = json.loads((run_dir / "build.json").read_text())
    scratch = Path(tempfile.mkdtemp(prefix="miles-pr-image-", dir=manifest["scratch"]))
    scratch.chmod(0o700)
    builder = "miles-pr-image-" + scratch.name.removeprefix("miles-pr-image-")
    env = {
        "PATH": PATH,
        "HOME": str(scratch / "home"),
        "DOCKER_CONFIG": str(scratch / "docker"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "BUILDX_BUILDER": builder,
        "PYTHONNOUSERSITE": "1",
    }
    for name in ("home", "docker"):
        (scratch / name).mkdir(mode=0o700)
    code = 1
    created = False
    try:
        driver, context, python = prepare_node(manifest, scratch, env)
        created = True  # even a partially failed create must attempt cleanup
        subprocess.run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                builder,
                "--driver",
                "docker-container",
                "--buildkitd-flags",
                "--oci-worker-net=bridge",
                "--use",
            ],
            env=env,
            check=True,
        )
        command = list(manifest["command"])
        command[:2] = [str(python), str(driver / "docker/build.py")]
        command[command.index("--context") + 1] = str(context)
        subprocess.run(command, env=env, cwd=driver, check=True)
        code = 0
    finally:
        # No prune or GPU/test-container cleanup. Only this job's builder/volume.
        if created:
            result = subprocess.run(["docker", "buildx", "rm", builder], env=env)
            if result.returncode:
                code = 1
        shutil.rmtree(scratch)
        temporary = run_dir / "exit-code.tmp"
        temporary.write_text(str(code))
        temporary.replace(run_dir / "exit-code")


def prepare_node(manifest, scratch, env):
    driver = scratch / "trusted"
    (driver / "docker").mkdir(parents=True)
    for name in ("build.py", "image_inputs.py"):
        shutil.copyfile(Path(manifest["trusted"]) / "docker" / name, driver / "docker" / name)
    context = scratch / "source"
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            manifest["source"],
            str(context),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "checkout", "--detach", manifest["merge_sha"]],
        cwd=context,
        env=env,
        check=True,
    )
    spec = importlib.util.spec_from_file_location("spur_image_inputs", driver / "docker/image_inputs.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    if reader.compute(root=context, variant="rocm10-mi35x-pr") != manifest["inputs_hash"]:
        raise ValueError("Compute-node context does not match the frozen build inputs")
    venv = scratch / "venv"
    subprocess.run(["/usr/bin/python3", "-m", "venv", str(venv)], env=env, check=True)
    python = venv / "bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "typer"], env=env, check=True)
    return driver, context, python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "node", "cancel"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--merge-sha")
    parser.add_argument("--inputs-hash")
    parser.add_argument("--command")
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args.source.resolve(), args.merge_sha, args.inputs_hash, json.loads(args.command))
    elif args.mode == "node":
        node(args.run_dir.resolve())
    else:
        cancel(args.run_dir.resolve())


if __name__ == "__main__":
    main()
