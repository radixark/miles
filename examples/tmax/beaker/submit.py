"""Submit a TMax experiment config to Beaker, as open-instruct's mason.py does for its launch scripts.

    python examples/tmax/beaker/submit.py examples/tmax/configs/prod_1node.yaml
    python examples/tmax/beaker/submit.py examples/tmax/configs/smoke_sync.yaml --dry-run -- --num-rollout 1

Reads the config's ``beaker:`` section and writes a Beaker v2 spec: ``minRuntime`` + ``autoResume`` + task
retries (mason's ``--min_runtime --auto_resume --max_retries``; ``preemptible`` only without ``min_runtime``), the radixark/miles image, GPUs, shared memory,
weka mounts and the user's secrets. The job is pinned to the current git commit: the task clones this repository
at that commit and runs ``beaker/tmax_beaker_entry.sh <config>``, so a preempted job's retries, and edits made to
the checkout while it runs, cannot change what it trains. Arguments after ``--`` are launcher overrides.

Each submission is its own run: the config's ``exp_name`` gets a UTC date-time suffix (``<exp_name>_YYYYMMDD_HHMMSS``)
that names the checkpoint and trace directories, the W&B run and the Beaker experiment. Beaker's retries of that
submission keep the name, so they resume from its checkpoints; a later submission starts a new run.
``--exp-name NAME`` reuses an existing run's name exactly, to continue that run. ``--exclude-host HOST`` (repeatable)
keeps the job off a node, such as one with a full disk: the spec then lists the cluster's other nodes by hostname.

With ``nodes: N > 1`` the task gets N replicas the way mason's ``--num_nodes`` does (leader selection, host
networking, failure and preemption propagated to every replica, NCCL over InfiniBand); the entrypoint makes
replica 0 the Ray head and joins the others to it.

``beaker:`` keys: cluster, workspace, budget, priority, preemptible, min_runtime, retries, nodes, gpus (per node),
synchronized_start_timeout (multi-node: how long replicas wait for each other to start, default 1h),
shared_memory, image, sandbox_shards, mirror_url (``auto``: the running registry_mirror.py mirrors), keep_last_n_checkpoints, weka (list of buckets, mounted at
/weka/<bucket>), dockerhub_username, secrets (env var -> Beaker secret), env (extra env vars); with the config's
``sandbox_backend: gateway``: gateway_registry (the podman fleet's head registry, for a gateway started on the head)
or gateway_url (a running gateway), gateway_min_replicas, literegistry_version, literegistry_podman_client_version.
"""

import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tmax_wandb  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ENTRY = "examples/tmax/beaker/tmax_beaker_entry.sh"
_SNAPSHOT = "/tmp/miles_snapshot"


def _description(*, config: dict, config_stem: str, exp_name: str, commit: str, overrides: list[str]) -> str:
    """The experiment's Beaker description: config, run name, commit, and the W&B run's URL when it is known.

    The URL needs the entity and project (the config's wandb_entity / wandb_project, or their launcher overrides) and
    a run name; the launcher logs the run under the same id (tmax_wandb.run_id).
    """
    text = f"Miles TMax {config_stem} {exp_name} @ {commit[:9]}".replace("  ", " ")
    wandb = {key: config.get(key, "") for key in ("wandb_entity", "wandb_project")}
    for flag, key in (("--wandb-entity", "wandb_entity"), ("--wandb-project", "wandb_project")):
        if flag in overrides[:-1]:
            wandb[key] = overrides[overrides.index(flag) + 1]
    if exp_name and all(wandb.values()):
        text += " | W&B " + tmax_wandb.run_url(
            entity=wandb["wandb_entity"], project=wandb["wandb_project"], exp_name=exp_name
        )
    return text


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    beaker = config.get("beaker") or {}
    commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--", "examples/tmax") and not args.allow_dirty:
        raise SystemExit("examples/tmax has uncommitted changes; the job runs the committed config. Commit first.")
    exp_name = args.exp_name or (
        f"{config['exp_name']}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}" if config.get("exp_name") else ""
    )
    name = args.name or exp_name or f"{config_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    spec = _build_spec(
        beaker=beaker,
        config_rel=str(config_path.relative_to(_REPO)),
        commit=commit,
        overrides=args.overrides,
        exclude_hosts=args.exclude_host,
        exp_name=exp_name,
        description=_description(
            config=config, config_stem=config_path.stem, exp_name=exp_name, commit=commit, overrides=args.overrides
        ),
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
        usage="%(prog)s config [--exp-name NAME] [--name NAME] [--exclude-host HOST] [--dry-run] [--allow-dirty] "
        "[-- launcher flags]",
    )
    parser.add_argument("config")
    parser.add_argument("--exp-name", default="", help="continue this existing run (no date-time suffix is added)")
    parser.add_argument("--name", default="", help="Beaker experiment name (default: the run name)")
    parser.add_argument(
        "--exclude-host",
        action="append",
        default=[],
        help="keep the job off this node (repeatable), e.g. holmes-cs-aus-556",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="submit although examples/tmax is uncommitted")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.overrides = argv[split + 1 :]
    return args


def _build_spec(
    *,
    beaker: dict,
    config_rel: str,
    commit: str,
    overrides: list[str],
    exclude_hosts: list[str],
    exp_name: str,
    description: str,
) -> dict:
    user = getpass.getuser().lower()
    secrets = beaker.get("secrets") or {
        "DOCKER_PAT": f"{user}_DOCKER_PAT",
        "HF_TOKEN": f"{user}_HF_TOKEN",
        "WANDB_API_KEY": f"{user}_WANDB_API_KEY",
    }
    nodes = int(beaker.get("nodes", 1))
    env = {
        "NUM_NODES": str(nodes),
        "BEAKER_ALLOW_SUBCONTAINERS": "1",
        "BEAKER_SKIP_DOCKER_SOCKET": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "DOCKERHUB_USERNAME": beaker.get("dockerhub_username", os.environ.get("DOCKERHUB_USERNAME", "")),
        "SANDBOX_SHARDS": str(beaker.get("sandbox_shards", 1)),
        "MIRROR_URL": _mirror_url(beaker.get("mirror_url", "")),
        "KEEP_LAST_N_CHECKPOINTS": str(beaker.get("keep_last_n_checkpoints", 0)),
        "GATEWAY_REGISTRY": beaker.get("gateway_registry", ""),
        "GATEWAY_URL": beaker.get("gateway_url", ""),
        "GATEWAY_MIN_REPLICAS": str(beaker.get("gateway_min_replicas", "")),
        "LITEREGISTRY_VERSION": str(beaker.get("literegistry_version", "")),
        "LITEREGISTRY_PODMAN_CLIENT_VERSION": str(beaker.get("literegistry_podman_client_version", "")),
        "EXTRA_ARGS": shlex.join(overrides),
        # The launcher's exported settings win over the config file's, and the entrypoint's pruner reads it too.
        "MILES_SCRIPT_EXP_NAME": exp_name,
        # mason's multi-node defaults on the weka clusters: NCCL over the InfiniBand NICs, not the bonded Ethernet.
        **({"NCCL_SOCKET_IFNAME": "ib", "NCCL_IB_HCA": "^=mlx5_bond_0"} if nodes > 1 else {}),
        **{key: str(value) for key, value in (beaker.get("env") or {}).items()},
    }
    git_dir = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    command = (
        "git config --global --add safe.directory '*' && "
        f"rm -rf {_SNAPSHOT} && git clone -q --shared {git_dir} {_SNAPSHOT} && "
        f"git -C {_SNAPSHOT} checkout -q {commit} && "
        f"exec bash {_SNAPSHOT}/{_ENTRY} {config_rel}"
    )
    # Beaker rejects preemptible together with minRuntime / autoResume: a min-runtime job is preemptible once that
    # time has passed and resumes automatically, which is mason's --min_runtime --auto_resume.
    context = {"priority": beaker.get("priority", "normal")}
    if beaker.get("min_runtime"):
        context |= {"minRuntime": beaker["min_runtime"], "autoResume": True}
    elif beaker.get("preemptible"):
        context |= {"preemptible": True}
    # mason --num_nodes: replica 0 is the leader (the Ray head), and one replica failing or being preempted stops
    # them all, so a retry restarts the whole cluster.
    replicas = (
        {
            "replicas": nodes,
            "leaderSelection": True,
            "hostNetworking": True,
            "propagateFailure": True,
            "propagatePreemption": True,
            # Beaker cancels the other replicas when one has not started within this window (default 10 min) and
            # does not retry the cancelled ones, which leaves a retried job with part of its nodes for good. A
            # replica whose node fails Beaker's health check needs longer than 10 min to be placed elsewhere.
            "synchronizedStartTimeout": beaker.get("synchronized_start_timeout", "1h"),
        }
        if nodes > 1
        else {}
    )
    return {
        "version": "v2",
        "description": description,
        "budget": beaker.get("budget", "ai2/oe-omai"),
        "tasks": [
            {
                "name": "miles-tmax",
                **replicas,
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
                "constraints": _constraints(beaker.get("cluster", "ai2/holmes"), exclude_hosts),
            }
        ],
        "retry": {"allowedTaskRetries": int(beaker.get("retries", 0))},
    }


def _constraints(cluster: str, exclude_hosts: list[str]) -> dict:
    """The cluster, or with ``exclude_hosts`` its other nodes by hostname: Beaker can pin hostnames but not exclude
    them, and takes a cluster or hostnames, not both."""
    if not exclude_hosts:
        return {"cluster": [cluster]}
    out = subprocess.run(["beaker", "cluster", "nodes", cluster, "--format", "json"], check=True, capture_output=True)
    hostnames = sorted(node["hostname"] for node in json.loads(out.stdout) if not node.get("cordoned"))
    excluded = {host.split(".")[0] for host in exclude_hosts}
    unknown = excluded - {hostname.split(".")[0] for hostname in hostnames}
    if unknown:
        raise SystemExit(f"--exclude-host {sorted(unknown)}: not a node of {cluster}")
    return {"hostname": [hostname for hostname in hostnames if hostname.split(".")[0] not in excluded]}


def _mirror_url(configured: str) -> str:
    """``auto``: the mirrors running now (registry_mirror.py urls), resolved at submit time like open-instruct's
    hand-copied MIRROR_URL, so a retry keeps them even after they move; podman then falls back to docker.io."""
    if configured != "auto":
        return configured
    script = _REPO / "examples/tmax/beaker/registry_mirror/registry_mirror.py"
    urls = subprocess.run([sys.executable, str(script), "urls"], check=True, capture_output=True, text=True)
    if not urls.stdout.strip():
        print("warning: no registry mirror is running; sandbox images will come from docker.io", file=sys.stderr)
    return urls.stdout.strip()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
