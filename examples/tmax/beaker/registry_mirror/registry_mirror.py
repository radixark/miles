"""Docker Hub pull-through mirrors for the TMax sandbox images, as open-instruct runs them.

    python examples/tmax/beaker/registry_mirror/registry_mirror.py launch            # 3 mirrors on ai2/neptune
    python examples/tmax/beaker/registry_mirror/registry_mirror.py urls              # MIRROR_URL for a config
    python examples/tmax/beaker/registry_mirror/registry_mirror.py probe             # /v2/ + a real task manifest
    python examples/tmax/beaker/registry_mirror/registry_mirror.py test              # pull-through + fallback job
    python examples/tmax/beaker/registry_mirror/registry_mirror.py warm              # one warm job per mirror

open-instruct's scripts/docker/launch_registry_mirror.sh (branch opd): a stock ``registry:3`` proxying
registry-1.docker.io with a Docker Hub login, host-networked, ``urgent`` with ``minRuntime`` 8h + ``autoResume``,
its cache on the node's disk. Two departures: the login comes from the user's Beaker secret through registry's
``REGISTRY_PROXY_*`` env overrides instead of a config file copied into a new secret, so rotating the token is one
secret update; and every mirror has its own port pair (registry ``port``, metrics ``port + 1``), since host
networking would otherwise collide two mirrors on one node. A mirror's address is ``<node hostname>:<port>``,
which moves when Beaker reschedules it (with an empty cache): ``urls`` reads the current ones.

Jobs consume mirrors through ``MIRROR_URL`` (podman's ``[[registry.mirror]]`` list, tried in order before
docker.io itself, so a dead mirror only slows pulls down); the Beaker entrypoint shuffles the list per node to
spread the load. ``warm_tmax_registry_mirror.py`` is open-instruct's warmer from opd @ 1bf48f1a7, only
reformatted to this repo's style (one wrapped line, an unused import dropped).
"""

import argparse
import getpass
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

_NAME = "tmax-registry-mirror"
_REPO = Path(__file__).resolve().parents[4]
_HERE = Path(__file__).resolve().parent.relative_to(_REPO)
_SNAPSHOT = "/tmp/miles_snapshot"
# A task image of allenai/tmax-15k-open-instruct: a liveness check on /v2/ alone misses a mirror serving corrupt
# manifests (open-instruct docs/registry_mirror.md), so probes fetch a real one.
_PROBE_IMAGE = "hamishi740/swerl-tmax-v3"
_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["launch", "urls", "probe", "test", "warm"])
    parser.add_argument("--ports", type=int, nargs="+", default=[5000, 5010, 5020], help="one mirror per port")
    parser.add_argument("--cluster", default="ai2/neptune")
    parser.add_argument("--workspace", default="ai2/oe-agents-holmes")
    parser.add_argument("--budget", default="ai2/oe-omai")
    parser.add_argument("--dockerhub-username", default="shashankg209")
    parser.add_argument("--docker-pat-secret", default=f"{getpass.getuser().lower()}_DOCKER_PAT")
    parser.add_argument("--hf-token-secret", default=f"{getpass.getuser().lower()}_HF_TOKEN")
    parser.add_argument("--image", default="docker.io/radixark/miles:dev-202610151741", help="for test / warm jobs")
    parser.add_argument("--warm-workers", type=int, default=8)
    parser.add_argument("--warm-args", default="", help="extra warm_tmax_registry_mirror.py flags (--warm-args=...)")
    args = parser.parse_args()
    {"launch": _launch, "urls": _print_urls, "probe": _probe, "test": _test, "warm": _warm}[args.command](args)


def _launch(args: argparse.Namespace) -> None:
    running = {port for port, _ in _mirrors(args)}
    for port in args.ports:
        if port in running:
            print(f"mirror on port {port} already running")
            continue
        _create(args, name=f"{_NAME}-{port}-{time.strftime('%Y%m%d-%H%M%S')}", spec=_mirror_spec(args, port))


def _mirror_spec(args: argparse.Namespace, port: int) -> dict:
    env = {
        "REGISTRY_HTTP_ADDR": f":{port}",
        "REGISTRY_HTTP_DEBUG_ADDR": f":{port + 1}",
        "REGISTRY_LOG_LEVEL": "info",
        "REGISTRY_PROXY_REMOTEURL": "https://registry-1.docker.io",
        "REGISTRY_PROXY_USERNAME": args.dockerhub_username,
    }
    return {
        "version": "v2",
        "description": f"Docker Hub pull-through cache for TMax sandboxes on port {port}",
        "budget": args.budget,
        "tasks": [
            {
                "name": "registry-mirror",
                "image": {"docker": "docker.io/library/registry:3"},
                "envVars": [{"name": k, "value": v} for k, v in env.items()]
                + [{"name": "REGISTRY_PROXY_PASSWORD", "secret": args.docker_pat_secret}],
                "hostNetworking": True,
                "resources": {"cpuCount": 8, "memory": "32GiB"},
                "context": {"priority": "urgent", "minRuntime": "8h", "autoResume": True},
                "constraints": {"cluster": [args.cluster]},
            }
        ],
    }


def _mirrors(args: argparse.Namespace) -> list[tuple[int, str]]:
    """(port, host:port) of every running mirror experiment."""
    listing = _beaker_json("workspace", "experiments", args.workspace, "--text", _NAME)
    found = []
    for experiment in listing:
        name = experiment.get("name", "")
        jobs = _beaker_json("experiment", "get", experiment["id"])
        jobs = (jobs[0] if isinstance(jobs, list) else jobs).get("jobs") or []
        status = (jobs[-1].get("status") or {}) if jobs else {}
        if not name.startswith(f"{_NAME}-") or not status.get("started") or status.get("exited"):
            continue
        node = _beaker_json("node", "get", jobs[-1]["node"])
        hostname = (node[0] if isinstance(node, list) else node)["hostname"]
        port = int(name.split("-")[3])
        found.append((port, f"{hostname}:{port}"))
    return sorted(found)


def _print_urls(args: argparse.Namespace) -> None:
    print(",".join(url for _, url in _mirrors(args)))


def _probe(args: argparse.Namespace) -> None:
    tag = _probe_tag()
    for _, url in _mirrors(args):
        print(url, _get(f"http://{url}/v2/"), _get(f"http://{url}/v2/{_PROBE_IMAGE}/manifests/{tag}"))


def _probe_tag() -> str:
    """A task image tag, read from the dataset's prompt JSONL that `prepare` writes."""
    prompts = Path("/weka/nora-default/shashankg/miles_tmax_v2/datasets/tmax-15k-open-instruct/miles_prompts.jsonl")
    with prompts.open() as f:
        return json.loads(f.readline())["metadata"]["image"].rsplit(":", 1)[1]


def _get(url: str) -> str:
    start = time.monotonic()
    request = urllib.request.Request(url, headers={"Accept": _MANIFEST_ACCEPT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
            return f"{response.status} in {time.monotonic() - start:.1f}s"
    except Exception as e:
        return f"FAILED ({e!r})"[:160]


def _test(args: argparse.Namespace) -> None:
    mirrors = ",".join(url for _, url in _mirrors(args))
    assert mirrors, "no mirror is running"
    script = f"bash {_SNAPSHOT}/{_HERE}/test_pull_through.sh {mirrors} docker.io/{_PROBE_IMAGE}:{_probe_tag()}"
    _create(args, name=f"{_NAME}-test-{time.strftime('%Y%m%d-%H%M%S')}", spec=_job_spec(args, script, cpus=4))


def _warm(args: argparse.Namespace) -> None:
    for port, url in _mirrors(args):
        if port not in args.ports:
            continue
        script = (
            f"pip install -q tqdm && python {_SNAPSHOT}/{_HERE}/warm_tmax_registry_mirror.py "
            f"--dataset allenai/tmax-15k-open-instruct --mirror {url} --workers {args.warm_workers} {args.warm_args}"
        )
        _create(args, name=f"{_NAME}-warm-{port}-{time.strftime('%Y%m%d-%H%M%S')}", spec=_job_spec(args, script))


def _job_spec(args: argparse.Namespace, script: str, cpus: int = 16) -> dict:
    """A CPU job on the mirrors' cluster running ``script`` from this commit's checkout."""
    commit = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    git_dir = f"{_REPO}/.git"
    command = (
        "git config --global --add safe.directory '*' && "
        f"rm -rf {_SNAPSHOT} && git clone -q --shared {git_dir} {_SNAPSHOT} && "
        f"git -C {_SNAPSHOT} checkout -q {commit.stdout.strip()} && {script}"
    )
    return {
        "version": "v2",
        "description": script[:200],
        "budget": args.budget,
        "tasks": [
            {
                "name": "main",
                "image": {"docker": args.image},
                "command": ["bash", "-c", command],
                "datasets": [{"mountPath": "/weka/nora-default", "source": {"weka": "nora-default"}}],
                "envVars": [
                    {"name": "BEAKER_ALLOW_SUBCONTAINERS", "value": "1"},
                    {"name": "BEAKER_SKIP_DOCKER_SOCKET", "value": "1"},
                    {"name": "DOCKERHUB_USERNAME", "value": args.dockerhub_username},
                    {"name": "HF_HOME", "value": "/weka/nora-default/shashankg/miles_tmax_v2/hf_home"},
                    {"name": "DOCKER_PAT", "secret": args.docker_pat_secret},
                    {"name": "HF_TOKEN", "secret": args.hf_token_secret},
                ],
                "resources": {"cpuCount": cpus, "memory": "64GiB"},
                "context": {"priority": "urgent", "preemptible": True},
                "constraints": {"cluster": [args.cluster]},
            }
        ],
    }


def _create(args: argparse.Namespace, *, name: str, spec: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    subprocess.run(
        ["beaker", "experiment", "create", "--workspace", args.workspace, "--name", name, f.name], check=True
    )


def _beaker_json(*argv: str):
    out = subprocess.run(["beaker", *argv, "--format", "json"], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


if __name__ == "__main__":
    main()
