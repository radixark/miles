"""Submit a multi-node ``--cluster-backend monarch`` run to SLURM, as one Monarch ``SlurmJob`` in batch mode.

    python -m miles.utils.workers.monarch.slurm --nodes 2 --gpus-per-node 8 \\
        --slurm-arg=--partition=gpu --slurm-arg=--time=4:00:00 -- \\
        python scripts/run_qwen3_4b.py train --num-gpus-per-node 8

The job runs the launcher command after ``--`` unchanged, as the batch script on SLURM's head node (the first node
of the allocation), with ``MILES_SCRIPT_CLUSTER_BACKEND=monarch``. Monarch starts a worker loop on every node, the
driver attaches to them as its worker hosts, and the allocation is released once the launcher command exits.

Nothing is copied to the nodes: every node runs this Python by its full path, from this directory, with this
environment (SLURM's default ``--export=ALL``), so the venv, the checkout and the run's paths must resolve to the
same files on every node, e.g. on a shared network filesystem. ``--log-dir`` must be shared too: the job reads the
job state that the submission saves there.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import logging
import os
import shlex
import sys
import uuid
from dataclasses import dataclass

from monarch.job import DEFAULT_JOB_PATH, SlurmJob

from miles.utils.logging_utils import configure_logger_raw
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX
from miles.utils.workers.monarch.env_vars import MONARCH_JOB_STATE_ENV_VAR
from miles.utils.workers.monarch.hosts import WORKER_MESH_NAME
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)

_CLUSTER_BACKEND_ENV_VAR = f"{SCRIPT_ENV_VAR_PREFIX}CLUSTER_BACKEND"
_DEFAULT_MONARCH_PORT = 22222
# sbatch flags this submission or SlurmJob already sets, or that would change the tasks and gpus the job's worker
# loops get; a second value in --slurm-arg would conflict with it
_RESERVED_SBATCH_FLAGS = frozenset(
    {
        "--nodes",
        "-N",
        "--ntasks-per-node",
        "--ntasks",
        "-n",
        "--gpus-per-node",
        "--gpus",
        "-G",
        "--gres",
        "--array",
        "-a",
        "--wrap",
        "--job-name",
        "-J",
        "--output",
        "-o",
        "--error",
        "-e",
        "--exclusive",
        "--chdir",
        "-D",
        "--export",
        "--parsable",
        "--quiet",
        "-Q",
    }
)


@dataclass(frozen=True, kw_only=True)
class SlurmSubmission:
    """One job to submit, checked as it is built so a bad one fails before anything is submitted."""

    # the launcher command the job runs as its batch script on the head node
    command: list[str]
    num_nodes: int
    # the gpus the job requests on each node
    gpus_per_node: int
    # extra sbatch flags, one per entry, e.g. "--partition=gpu"; never one the submission sets itself
    slurm_args: list[str]
    # a directory every node sees, which the job's logs and saved state go under
    log_dir: str
    # the prefix of the job's name, and of its directory under log_dir
    job_name: str
    # whole nodes; without it the job shares its nodes, and --slurm-arg has to ask for cpus and memory
    exclusive: bool = True
    # the address the worker loops listen on; None listens on what the node's hostname resolves to there
    bind_to: str | None = None
    # the port the worker loops listen on, which two jobs sharing a node must not both take
    monarch_port: int = _DEFAULT_MONARCH_PORT

    def __post_init__(self) -> None:
        assert self.command, "pass the launcher command to run in the job after --"
        assert self.num_nodes > 0 and self.gpus_per_node > 0, "--nodes and --gpus-per-node must be positive"
        if self.bind_to is not None:
            # SlurmJob would refuse it only after the submission made the job's directory
            ipaddress.ip_address(self.bind_to)
        for arg in self.slurm_args:
            # SlurmJob writes only an entry that starts with a dash, as one "#SBATCH <entry>" line, and drops the rest
            assert arg.startswith("-"), f"--slurm-arg {arg!r} is no sbatch flag, which sbatch would never see"
        # one entry may hold several options
        for token in (token for arg in self.slurm_args for token in shlex.split(arg)):
            flag = token.split("=", 1)[0]
            assert not (
                flag.startswith("-") and (flag in _RESERVED_SBATCH_FLAGS or flag[:2] in _RESERVED_SBATCH_FLAGS)
            ), f"--slurm-arg must not set {flag}, which the submission already sets"


def check_submitting_environment() -> None:
    """The job runs with this shell's environment, so refuse what would mislead it."""
    export_mode = (os.environ.get("SBATCH_EXPORT") or "ALL").split(",")[0]
    assert export_mode == "ALL", "the job runs with the submitting environment, so SBATCH_EXPORT must be unset or ALL"
    # the job would inherit it, although which node is the run's head node is only known once the job runs
    assert "MASTER_ADDR" not in os.environ, "MASTER_ADDR is set in the submitting environment; unset it"


def submit(submission: SlurmSubmission) -> str:
    """Submit the job and return its SLURM job id."""
    check_submitting_environment()
    job_dir = os.path.join(os.path.abspath(submission.log_dir), f"{submission.job_name}-{uuid.uuid4().hex[:6]}")
    os.makedirs(job_dir)
    job = SlurmJob(
        meshes={WORKER_MESH_NAME: submission.num_nodes},
        python_exe=sys.executable,
        slurm_args=[f"--chdir={os.getcwd()}", *submission.slurm_args],
        job_name=submission.job_name,
        log_dir=job_dir,
        exclusive=submission.exclusive,
        gpus_per_node=submission.gpus_per_node,
        bind_to=submission.bind_to,
        monarch_port=submission.monarch_port,
    )
    # SlurmJob.apply saves the job state to DEFAULT_JOB_PATH, relative to the directory it runs in; Monarch's batch
    # runner splits the client script and runs it without a shell, so env sets what the job adds to this environment
    run_env = {
        _CLUSTER_BACKEND_ENV_VAR: ClusterBackend.MONARCH.value,
        MONARCH_JOB_STATE_ENV_VAR: os.path.join(job_dir, DEFAULT_JOB_PATH),
    }
    client_script = shlex.join(["env", *(f"{name}={value}" for name, value in run_env.items()), *submission.command])
    logger.info(f"Submitting {submission.num_nodes} nodes running: {client_script}")
    with contextlib.chdir(job_dir):
        job.apply(client_script=client_script)
    logger.info(f"SLURM logs and the job state are in {job_dir}")
    assert (job_id := job.job_id) is not None, "SlurmJob.apply returned without a SLURM job id"
    return job_id


def _parse_args(argv: list[str]) -> tuple[SlurmSubmission, bool]:
    """The submission the command line describes, and whether it is a dry run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, required=True, help="Nodes of the job; the driver runs on the first.")
    parser.add_argument("--gpus-per-node", type=int, required=True, help="GPUs the job requests on each node.")
    parser.add_argument(
        "--slurm-arg",
        action="append",
        default=[],
        help="One sbatch flag, e.g. --slurm-arg=--partition=gpu; repeat it for more. SBATCH_* variables override them.",
    )
    parser.add_argument(
        "--no-exclusive",
        action="store_true",
        help="Share the nodes with other jobs instead of reserving them whole; ask for cpus and memory with "
        "--slurm-arg=--cpus-per-task=... --slurm-arg=--mem=..., and the cluster must confine each job to its gpus.",
    )
    parser.add_argument(
        "--bind-to",
        default=None,
        help="Address the worker loops listen on, e.g. :: on a dual-stack cluster, where a node's hostname can "
        "resolve to IPv6 on the node itself and to IPv4 on the driver's node.",
    )
    parser.add_argument(
        "--monarch-port",
        type=int,
        default=_DEFAULT_MONARCH_PORT,
        help="Port the worker loops listen on; with --no-exclusive, pick one no other job on the nodes uses.",
    )
    parser.add_argument("--log-dir", default="monarch_slurm", help="Shared directory for the job's logs and state.")
    parser.add_argument("--job-name", default="miles", help="Prefix of the job's name.")
    parser.add_argument("--dry-run", action="store_true", help="Print the submission without submitting it.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- and the launcher command the job runs.")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    submission = SlurmSubmission(
        command=command,
        num_nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        slurm_args=args.slurm_arg,
        log_dir=args.log_dir,
        job_name=args.job_name,
        exclusive=not args.no_exclusive,
        bind_to=args.bind_to,
        monarch_port=args.monarch_port,
    )
    return submission, args.dry_run


def main(argv: list[str] | None = None) -> str | None:
    """Return the SLURM job id, or None on --dry-run."""
    configure_logger_raw("launcher")
    submission, dry_run = _parse_args(sys.argv[1:] if argv is None else argv)
    print(submission, flush=True)
    if dry_run:
        check_submitting_environment()
        return None
    job_id = submit(submission)
    print(f"Submitted SLURM job {job_id}", flush=True)
    return job_id


if __name__ == "__main__":
    main()
