import subprocess
from collections.abc import Sequence

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

KUBECTL_TIMEOUT_SECONDS: float = 60.0
LEADER_WORKER_SET_KIND: str = "leaderworkersets.leaderworkerset.x-k8s.io"


def read_objects_of_release(
    *, kind: str, release: str, namespace: str, output: str, extra_labels: Sequence[str] = ()
) -> str:
    result = _run_kubectl(
        [
            "get",
            kind,
            "--namespace",
            namespace,
            "--selector",
            compute_release_selector(release=release, extra_labels=extra_labels),
            "--output",
            output,
        ]
    )
    return result.stdout


def read_replicas(*, namespace: str, workload: str) -> int:
    result = _run_kubectl(
        ["get", LEADER_WORKER_SET_KIND, workload, "--namespace", namespace, "-o", "jsonpath={.spec.replicas}"]
    )
    return int(result.stdout.strip())


def compute_release_selector(*, release: str, extra_labels: Sequence[str] = ()) -> str:
    return ",".join([f"{INSTANCE_LABEL}={release}", *extra_labels])


def _run_kubectl(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_process(["kubectl", *args], capture_output=True, check=check, timeout=KUBECTL_TIMEOUT_SECONDS)
