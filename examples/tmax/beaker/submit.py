"""Submit a TMax experiment config to Beaker, as open-instruct's mason.py does for its launch scripts.

    python examples/tmax/beaker/submit.py examples/tmax/configs/prod_1node.yaml
    python examples/tmax/beaker/submit.py examples/tmax/configs/smoke_sync.yaml --dry-run -- --num-rollout 1

Reads the config's ``beaker:`` section and writes a Beaker v2 spec: ``preemptible`` + ``autoResume`` + task
retries (mason's ``--preemptible --auto_resume --max_retries``), the radixark/miles image, GPUs, shared memory,
weka mounts and the user's secrets. The job is pinned to the current git commit: the task clones this repository
at that commit and runs ``beaker/tmax_beaker_entry.sh <config>``, so a preempted job's retries, and edits made to
the checkout while it runs, cannot change what it trains. Arguments after ``--`` are launcher overrides.

``beaker:`` keys: cluster, workspace, budget, priority, preemptible, min_runtime, retries, gpus, shared_memory,
image, sandbox_shards, mirror_url, keep_last_n_checkpoints, weka (list of buckets, mounted at /weka/<bucket>),
dockerhub_username, secrets (env var -> Beaker secret), env (extra env vars).
"""

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
_ENTRY = "examples/tmax/beaker/tmax_beaker_entry.sh"
_SNAPSHOT = "/tmp/miles_snapshot"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    beaker = config.get("beaker") or {}
    commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--", "examples/tmax") and not args.allow_dirty:
        raise SystemExit("examples/tmax has uncommitted changes; the job runs the committed config. Commit first.")
    name = args.name or f"{config.get('exp_name') or config_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    spec = _build_spec(
        beaker=beaker,
        config_rel=str(config_path.relative_to(_REPO)),
        commit=commit,
        overrides=args.overrides,
        description=f"Miles TMax {config_path.stem} @ {commit[:9]}",
    )
    text = yaml.safe_dump(spec, sort_keys=False)
    if args.dry_run:
        print(text)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
    workspace = beaker.get("workspace", "ai2/oe-agents-holmes")
    subprocess.run(["beaker", "experiment", "create", "--workspace", workspace, "--name", name, f.name], check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s config [--name NAME] [--dry-run] [--allow-dirty] [-- launcher flags]",
    )
    parser.add_argument("config")
    parser.add_argument("--name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="submit although examples/tmax is uncommitted")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.overrides = argv[split + 1 :]
    return args


def _build_spec(*, beaker: dict, config_rel: str, commit: str, overrides: list[str], description: str) -> dict:
    user = getpass.getuser().lower()
    secrets = beaker.get("secrets") or {
        "DOCKER_PAT": f"{user}_DOCKER_PAT",
        "HF_TOKEN": f"{user}_HF_TOKEN",
        "WANDB_API_KEY": f"{user}_WANDB_API_KEY",
    }
    env = {
        "BEAKER_ALLOW_SUBCONTAINERS": "1",
        "BEAKER_SKIP_DOCKER_SOCKET": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "DOCKERHUB_USERNAME": beaker.get("dockerhub_username", os.environ.get("DOCKERHUB_USERNAME", "")),
        "SANDBOX_SHARDS": str(beaker.get("sandbox_shards", 1)),
        "MIRROR_URL": beaker.get("mirror_url", ""),
        "KEEP_LAST_N_CHECKPOINTS": str(beaker.get("keep_last_n_checkpoints", 0)),
        "EXTRA_ARGS": shlex.join(overrides),
        **{key: str(value) for key, value in (beaker.get("env") or {}).items()},
    }
    git_dir = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    command = (
        "git config --global --add safe.directory '*' && "
        f"rm -rf {_SNAPSHOT} && git clone -q --shared {git_dir} {_SNAPSHOT} && "
        f"git -C {_SNAPSHOT} checkout -q {commit} && "
        f"exec bash {_SNAPSHOT}/{_ENTRY} {config_rel}"
    )
    context = {"priority": beaker.get("priority", "normal"), "preemptible": bool(beaker.get("preemptible", False))}
    if beaker.get("min_runtime"):
        context |= {"minRuntime": beaker["min_runtime"], "autoResume": True}
    return {
        "version": "v2",
        "description": description,
        "budget": beaker.get("budget", "ai2/oe-omai"),
        "tasks": [
            {
                "name": "miles-tmax",
                "image": {"docker": beaker.get("image", "docker.io/radixark/miles:latest")},
                "command": ["bash", "-c", command],
                "datasets": [
                    {"mountPath": f"/weka/{bucket}", "source": {"weka": bucket}}
                    for bucket in beaker.get("weka", ["nora-default", "oe-adapt-default"])
                ],
                "envVars": [{"name": k, "value": v} for k, v in env.items() if v != ""]
                + [{"name": k, "secret": v} for k, v in secrets.items()],
                "resources": {
                    "gpuCount": int(beaker["gpus"]),
                    "sharedMemorySize": beaker.get("shared_memory", "256GiB"),
                },
                "context": context,
                "constraints": {"cluster": [beaker.get("cluster", "ai2/holmes")]},
            }
        ],
        "retry": {"allowedTaskRetries": int(beaker.get("retries", 0))},
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
